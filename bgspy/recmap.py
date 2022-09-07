import warnings
from collections import namedtuple, defaultdict
from scipy import interpolate
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

    Loads a HapMap or BED recombination map and builds rate and cummulative
    rate interpolators. Note that a conversion_factor of 1e-8 is the default
    for cM/Mb rates (1cM = 0.01M, 1Mb = 10^6 bases, 0.01 / 1e6 = 1e-8).

    Notes:
     Linear interpolation is the default -- quadratic can lead to some
     rather strange values. One should randomly sample positions and
     verify the cummulative interpolation is working proplerly on
     new maps.

    """
    def __init__(self, mapfile, seqlens, fill_to_end=True,
                 cumm_interpolation='linear',
                 rate_interpolation='linear',
                 conversion_factor=1e-8):
        self.mapfile = mapfile
        self.conversion_factor = conversion_factor
        self.ends = dict()
        self.rates = None
        self._fill = fill_to_end
        assert isinstance(seqlens, dict), "seqlens must be a dict"
        self.seqlens = seqlens
        self.cumm_rates = None
        self.params = []
        self.cumm_interpolation = cumm_interpolation
        self.rate_interpolation = rate_interpolation
        self._readmap()

    def __repr__(self):
        tot = sum([x.rate[-1] for x in self.cumm_rates.values()])
        file = self.mapfile
        return f"RecMap('{file}')\n total length: {np.round(tot, 2)} Morgans"

    def _readmap(self):
        """
        Notes:
         cummulative rates are calculated as:
          |  r1  |    r2       |         r3     |
          |  w1  |    w2       |         w3     |
                 m1            m2              m3
        c1: = w1*r1
        c2: = w2*r2 + w1*r1
        c3: = w3*r3 + w2*r2 + w1*r1
        """
        rates = defaultdict(list)
        last_chrom, last_end = None, None
        first_bin = True
        first = True
        is_hapmap = False
        ignored_chroms = set()
        with readfile(self.mapfile) as f:
            for line in f:
                if line.startswith('Chromosome'):
                    print(f"ignoring HapMap header...")
                    is_hapmap = True
                    continue
                if line.startswith('#'):
                    self.params.append(line.strip().lstrip('#'))
                cols = line.strip().split("\t")
                is_hapmap = is_hapmap or len(cols) == 3
                if is_hapmap:
                    if first:
                        print("parsing recmap as HapMap formatted (chrom, end, rate)")
                        first = False
                    chrom, end, rate = line.strip().split("\t")[:3]
                    start = -1
                else:
                    # BED file version
                    if first:
                        print("parsing recmap as BED formatted (chrom, start, end, rate)")
                        first = False
                    chrom, start, end, rate = line.strip().split("\t")[:4]
                if chrom not in self.seqlens:
                    ignored_chroms.add(chrom)
                    continue
                if last_chrom is not None and chrom != last_chrom:
                    # propagate the ends list
                    self.ends[last_chrom] = last_end
                    first_bin = True
                if first_bin and start != 0:
                    # missing data up until this point, fill in with an nan
                    start = start if not is_hapmap else 0
                    rates[chrom].append((int(start), np.nan))
                    first_bin = False
                rates[chrom].append((int(end), float(rate)))
                last_chrom = chrom
                last_end = int(end)

        self.ends[last_chrom] = last_end

        if len(ignored_chroms):
            print(f"RecMap._readmap() ignored {', '.join(ignored_chroms)}")

        if self._fill:
            for chrom in rates.keys():
                not_to_end = rates[chrom][-1][0] < self.seqlens[chrom]
                if not_to_end:
                    rates[chrom].append((self.seqlens[chrom], 0.0))

        cumm_rates = dict()
        for chrom, data in rates.items():
            pos = np.array([p for p, _ in data])
            rate = np.array([r for _, r in data])
            rbp = rate * self.conversion_factor
            rates[chrom] = RecPair(pos, rbp)
            assert pos[0] == 0
            # note: we exclude the first bin (0) -- the cummulative
            # distances should start from the first marker onwards (otherwise
            # there's a jump)
            widths = np.diff(pos[1:])
            cumrates = np.nancumsum(rbp[1:-1]*widths)
            cumm_pos = pos[1:-1]
            assert len(cumm_pos) == len(cumrates)
            cumm_rates[chrom] = RecPair(cumm_pos, cumrates)
        self.rates = rates
        self.cumm_rates = cumm_rates
        self.cumm_interpol = rate_interpol(cumm_rates, kind=self.cumm_interpolation)
        self.inverse_cumm_interpol = rate_interpol(cumm_rates, inverse=True,
                                                   kind=self.cumm_interpolation)
        self.rate_interpol = rate_interpol(rates, kind=self.rate_interpolation)

    def lookup(self, chrom, pos, cummulative=False):
        #assert(np.all(0 <= pos <= self.ends[chrom]))
        if np.any(pos > self.seqlens[chrom]):
            bad_pos = pos[pos > self.seqlens[chrom]]
            msg = f"some positions {bad_pos} are greater than sequence length ({self.seqlens[chrom]}"
            warnings.warn(msg)
        if not cummulative:
            x = self.rate_interpol[chrom](pos)
        else:
            x = self.cumm_interpol[chrom](pos)
        return x

    @property
    def map_lengths(self):
        return {chrom: x.rate[-1] for chrom, x in self.cumm_rates.items()}

    def build_recmaps(self, positions, cummulative=False):
        map_positions = defaultdict(list)
        for chrom in positions:
            for pos in positions[chrom]:
                map_positions[chrom].append(self.lookup(chrom, pos,
                                                        cummulative=cummulative))
            map_positions[chrom] = np.array(map_positions[chrom])
        return map_positions


