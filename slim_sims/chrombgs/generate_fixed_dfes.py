"""
Generate fixed DFEs.
"""
usage = """\
generate_fixed_dfes.py <prefix> <lower> <upper> <number>
"""

import sys
import numpy as np


try:
    prefix, lower, upper, number = sys.argv[1:]
except:
    sys.exit(usage)

sels = np.logspace(int(lower), int(upper), int(number))

def clean_float(x, digits=12):
    """
    just to clean up 0.99999.
    """
    return str(np.round(x, digits))

for sel in sels:
    with open(f"{prefix}_{clean_float(sel)}", 'w') as f:
        sel_str = ','.join(map(clean_float, sels))
        f.write(f"grid\t{sel_str}\n")
        dfe = np.zeros(len(sels))
        dfe[sels == sel] = 1.
        dfe_str = ','.join(map(clean_float, dfe))
        f.write(f"{prefix}\t{dfe_str}")
