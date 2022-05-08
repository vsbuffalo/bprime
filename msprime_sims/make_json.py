import json
import os

TEMPLATE = {
  "name": "varying_nsamples_{nsamples}_{nreps}",
  "model": "bgs_segment",
  "params":
     {
       "N": {"dist": {"name": "fixed", "val": 1000}, "type": "int"},
       "mu": {"dist": {"name": "log10_uniform", "low": -10, "high": -6}, "type": "float"},
       "s": {"dist": {"name": "log10_uniform", "low": -5, "high": -1}, "type": "float"},
       "h": {"dist": {"name": "fixed", "val": 0.5}, "type": "float"},
       "rf": {"dist": {"name": "log10_uniform", "low": -10, "high": -0.301}, "type": "float"},
       "rbp": {"dist": {"name":"log10_uniform", "low": -10, "high": -6}, "type": "float"},
       "L": {"dist": {"name": "log10_uniform", "low": -1, "high": 3.04}, "type": "float"}
     },
  "nsamples": None,
  "nreps": None
}

def fill_template(nsamples, nreps):
    template = TEMPLATE.copy()
    template['name'] = template['name'].format(nsamples=nsamples, nreps=nreps)
    template['nsamples'] = nsamples
    template['nreps'] = nreps
    return template


nsamples = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000]

nreps = [1, 2, 5, 10]

for nsample in nsamples:
    for nrep in nreps:
        with open(f"varying_nsamples_{nsample}_{nrep}.json", 'w') as f:
            json.dump(fill_template(nsample, nrep), f, indent="  ")