import warnings
import logging
from collections import namedtuple, defaultdict
from scipy import interpolate
import tskit as tsk
import numpy as np
import pandas as pd
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


def check_positions_sorted(pos):
    msg = "positions are not sorted!"
    assert np.all(np.diff(pos) >= 0), msg


def rates_to_cumulative(rates, pos):
    """
    Given end-to-end positions are marginal (per-basepair rates)
    compute the cumulative map distance at each position.
    """
    assert len(rates) == len(pos)-1
    spans = np.diff(pos)
    return np.array([0] + np.cumsum(rates * spans).tolist())


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
        file = self.mapfile
        is_four_col = len(open(file).readline().split('\t')) == 4
        three_col = ('chrom', 'pos', 'rate')
        four_col = ('chrom', 'pos', 'rate', 'cumrate')
        header = four_col if is_four_col else three_col
        has_header = open(file).readline().lower().startswith('chrom')
        d = pd.read_table(file, skiprows=int(has_header), names=header)
        new_rates = dict()
        ratemaps = dict()
        for chrom, df in d.groupby('chrom'):
            pos = df['pos'].tolist()
            rates = df['rate'].tolist()
            if is_four_col:
                cumrates = df['cumrate'].tolist()
                # currently we don't do anything with this yet but
                # it could be used for validation
            if pos[0] != 0:
                assert pos[0] > 0  # something is horribly wrong
                pos.insert(0, 0)
                # technically this is missing, but it's fine
                rates.insert(0, 0)
            if pos[-1] != self.seqlens[chrom]:
                # let's add the end in -- note this already has a rate!
                pos.append(self.seqlens[chrom])

            rates = np.array(rates)
            pos = np.array(pos)
            check_positions_sorted(pos)
            assert len(rates)+1 == len(pos)
            # TODO: check that rates and cumulative 
            # map positions all match
            new_rates[chrom] = (pos, rates)
            rates = self.conversion_factor*np.array(rates)
            # I use tskit to store the rate data and calculate the 
            # cumulative rate distances... it's robust and simplifies 
            # things
            ratemaps[chrom] = tsk.RateMap(position=pos, rate=rates)

        # all rates loaded in now, alias the name
        rates = new_rates

        #self._rm = rates  # we can store the ratemaps for debugging

        cum_rates = dict()
        for chrom, ratemap in ratemaps.items():
            cumrate = ratemap.get_cumulative_mass(ratemap.right)
            cum_rates[chrom] = RecPair(ratemap.right.astype(int),
                                       cumrate)

        self.rates = {c: RecPair(x.right.astype(int), x.rate) for c, x in ratemaps.items()}
        self.cum_rates = cum_rates
        self.cum_interpol = rate_interpol(cum_rates, kind=self.cum_interpolation)
        self.inverse_cum_interpol = rate_interpol(cum_rates, inverse=True,
                                                   kind=self.cum_interpolation)
        simple_rates = {c: (x.right, x.rate) for c, x in ratemaps.items()}
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


