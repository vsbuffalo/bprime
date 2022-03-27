import msprime

def load_recrates(file, seqlen, conversion_factor=1e-8):
    """
    Load BED file of recombination rate.
    """
    chroms = set()
    positions, rates = [0], [0.0]
    with open(file) as f:
        for line in f:
            chrom, start, end, rate = line.strip().split('\t')
            chroms.add(chrom)
            positions.append(int(end))
            rates.append(conversion_factor*float(rate))
    assert(len(chroms) == 1) # only one chromosome
    recmap = msprime.RateMap(position=positions, rate=rates[1:])
    recmap.chrom = list(chroms)[0]
    return recmap

def load_neutregions(file, rate, seqlen, ratemap=True):
    chroms = set()
    positions, rates = [], []
    last_end = None
    first = True
    with open(file) as f:
        for line in f:
            chrom, start, end = line.strip().split('\t')
            chroms.add(chrom)
            start, end = int(start), int(end)
            if first:
                positions.extend((start, end))
                rates.append(rate)
                first = False
                continue
            positions.append(start)
            rates.append(0)
            positions.append(end)
            rates.append(rate)
            last_end = end

    assert(len(chroms) == 1)
    if end < seqlen:
        positions.append(seqlen)
        rates.append(0)
    if not ratemap:
        return positions, rates
    ratemap =  msprime.RateMap(position=positions, rate=rates)
    ratemap.chrom = list(chroms)[0]
    return ratemap


