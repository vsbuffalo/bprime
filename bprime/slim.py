# slim.py -- helpers for snakemake/slim sims

def filename_pattern(base, params, seed=False, rep=False):
  param_str = [v + '{' + v + '}' for v in params]
  if seed:
    param_str.append('seed{seed}')
  if rep:
    param_str.append('rep{rep}')
  pattern = base + '_'.join(param_str) + '_{{output}}'
  return pattern

def slim_call(params, slim_cmd="slim", rep=False, seed=True, manual=None):
  call_args = []
  for p, (lower, upper, log10, type) in params.items():
    is_str = type is str
    val = f"{{wildcards.{p}}}" if not is_str else f'\\"{{wildcards.{p}}}\\"'
    call_args.append(f"-d {p}={val}")
  if rep:
    call_args.append("-d rep={wildcards.rep}")
  add_on = ''
  if manual is not None:
    # manual stuff
    add_on = []
    for key, val in manual.items():
      if isinstance(val, str):
        add_on.append(f'-d {key}=\\"{val}\\"')
      else:
        add_on.append(f'-d {key}={val}')
    add_on = ' ' + ' '.join(add_on)
  if seed:
    call_args.append("-s {wildcards.seed}")
  full_call = f"{slim_cmd} " + " ".join(call_args) + add_on
  return full_call

def param_grid(seed=False, **kwargs):
  params = []
  for param, values in kwargs.items():
    if len(values):
      params.append([(param, v) for v in values])
    else:
      params.append([(param, '')])
  out = list(map(dict, itertools.product(*params)))
  if not seed:
    return out
  for entry in out:
    entry['seed'] = np.random.randint(0, 2**63)
  return out


