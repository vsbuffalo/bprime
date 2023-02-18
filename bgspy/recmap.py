import warnings
from collections import namedtuple, defaultdict
from scipy import interpolate
import tskit as tsk
import numpy as np
from bgspy.utils import readfile, get_unique_indices

RecPair = namedtuple('RecPair', ('end', 'rate'))

def rate_interpol(rate_dict, inverse=False, **kwargs):
    """
    Interpolate recombination rates given a dictionary of chrom->(ends, rates).
    By default, it uses quadratic, and any values outside the range are given
    the two end points.

    If inverse=True, the rates and physical positions are inverted.
    """
    defaults = {'kind': 'quadratic',
                'assume_sorted': True,
                'bounds_error': False,
                'copy': False}
    kwargs = {**defaults, **kwargs}
    interpols = dict()
    for chrom, (x, y) in rate_dict.items():
        if inverse:
            y, x = x, y
            idx = get_unique_indices(x) # x's can't be duplicated
            x, y = x[idx], y[idx]
            ends = x[0], x[-1]
        else:
            ends = y[0], y[-1]
        interpol = interpolate.interp1d(x, y, fill_value=ends, **kwargs)
        interpols[chrom] = interpol
    return interpols


class RecMap(object):
    """
    Recombination Map Class

    Loads a HapMap or BED recombination map and builds rate and cumulative
    rate interpolators. Note that a conversion_factor of 1e-8 is the default
    for cM/Mb rates (1cM = 0.01M, 1Mb = 10^6 bases, 0.01 / 1e6 = 1e-8).

    Notes:
     Linear interpolation is the default -- quadratic can lead to some
     rather strange values. One should randomly sample positions and
     verify the cumulative interpolation is working proplerly on
     new maps.

    """
    def __init__(self, mapfile, seqlens, fill_to_end=True,
                 cum_interpolation='linear',
                 rate_interpolation='linear',
                 conversion_factor=1e-8):
        self.mapfile = mapfile
        self.conversion_factor = conversion_factor
        self.ends = dict()
        self.rates = None
        self._fill = fill_to_end
        assert isinstance(seqlens, dict), "seqlens must be a dict"
        self.seqlens = seqlens
        self.cum_rates = None
        self.params = []
        self.cum_interpolation = cum_interpolation
        self.rate_interpolation = rate_interpolation
        self.readmap()

    def __repr__(self):
        tot = sum([x.rate[-1] for x in self.cum_rates.values()])
        file = self.mapfile
        return f"RecMap('{file}')\n total length: {np.round(tot, 2)} Morgans"

    def readmap(self, header=True):
        """
        Notes:
         cumulative rates are calculated as:
          |  r1  |    r2       |         r3     |
          |  w1  |    w2       |         w3     |
                 m1            m2              m3
        c1: = w1*r1
        c2: = w2*r2 + w1*r1
        c3: = w3*r3 + w2*r2 + w1*r1
        """

        # check if has forth cumulative rate column
        has_cumrate = len(next(open(self.mapfile)).split('\t')) == 4
        cols = {'names': ['chrom', 'end', 'rate'],
                'formats': ['S5', 'int', 'float']}
        if has_cumrate:
            cols['names'].append('cum')
            cols['formats'].append('float')

        d = np.loadtxt(self.mapfile, skiprows=int(header), dtype=cols)
        raw_rates = defaultdict(list)
        for i in range(d.shape[0]):
            row = d[i]
            chrom, end, rate, *cum = row
            chrom = chrom.decode()
            raw_rates[chrom].append((end, rate))

        # go through and pre-prend zeros and append ends if needed
        rates = dict()
        for chrom, data in raw_rates.items():
            end = self.seqlens[chrom]
            if data[-1][0] > end:
                msg = f"{chrom} has an end passed the reference sequence!"
                raise ValueError(msg)
            #elif data[-1][0] < end:
                # NOTE: we put a None here since we need one fewer end rec rate
            #    data.append((end, 0))
            # else: it goes to the end, everything's good

            ends = [e for e, _ in data]
            rec_rates = [r for _, r in data]
            assert ends[0] != 0, " we can't a rate left of pos 0"
            ends.insert(0, 0)  # add in the left-most position
            # the hapmap format format is each line is rate between it and *next*
            # so popping in a zero means we don't know the rate -- fill with nan
            rec_rates.insert(0, 0)
            # if we don't go to end, we add that in -- note that the last
            # rate we have is to end.
            if ends[-1] < self.seqlens[chrom]:
                ends.append(self.seqlens[chrom])

            rec_rates = self.conversion_factor*np.array(rec_rates)
            rates[chrom] = tsk.RateMap(position=ends, rate=rec_rates)

        #self._rm = rates  # we can store the ratemaps for debugging

        cum_rates = dict()
        for chrom, ratemap in rates.items():
            cumrate = ratemap.get_cumulative_mass(ratemap.right)
            cum_rates[chrom] = RecPair(ratemap.right, cumrate)

        self.rates = {c: RecPair(x.right, x.rate) for c, x in rates.items()}
        self.cum_rates = cum_rates
        self.cum_interpol = rate_interpol(cum_rates, kind=self.cum_interpolation)
        self.inverse_cum_interpol = rate_interpol(cum_rates, inverse=True,
                                                   kind=self.cum_interpolation)
        simple_rates = {c: (x.right, x.rate) for c, x in rates.items()}
        self.rate_interpol = rate_interpol(simple_rates, kind=self.rate_interpolation)

    def lookup(self, chrom, pos, cumulative=False):
        #assert(np.all(0 <= pos <= self.ends[chrom]))
        if np.any(pos > self.seqlens[chrom]):
            bad_pos = pos[pos > self.seqlens[chrom]]
            msg = f"some positions {bad_pos} are greater than sequence length ({self.seqlens[chrom]}"
            warnings.warn(msg)
        if not cumulative:
            x = self.rate_interpol[chrom](pos)
        else:
            x = self.cum_interpol[chrom](pos)
        return x

    @property
    def map_lengths(self):
        return {chrom: x.rate[-1] for chrom, x in self.cum_rates.items()}

    def build_recmaps(self, positions, cumulative=False):
        map_positions = defaultdict(list)
        for chrom in positions:
            for pos in positions[chrom]:
                map_positions[chrom].append(self.lookup(chrom, pos,
                                                        cumulative=cumulative))
            map_positions[chrom] = np.array(map_positions[chrom])
        return map_positions


