import time
import os

total = 1e6

def get_nfiles():
  dir = 'fullbgs'
  nfiles = 0
  for root, dirs, files in os.walk(dir):
    nfiles += len([f for f in files if f.endswith("treeseq.tree")])
  return nfiles

t0 = time.time()
t1 = time.time()

nfiles0 = get_nfiles()
nfiles1 = get_nfiles()
while True:
    time.sleep(60)
    nfiles2 = get_nfiles()
    t2 = time.time()
    tdiff1 = t2 - t1
    tdiff0 = t2 - t0
    fdiff1 = nfiles2 - nfiles1
    fdiff0 = nfiles2 - nfiles0
    # reset
    nfiles1 = nfiles2
    t1 = t2
    tot_rate = fdiff0 / tdiff0
    inst_rate = fdiff1 / tdiff1
    nremain = total - nfiles2
    tremain = nremain / tot_rate
    print(f"total rate: {round(inst_rate, 1)} jobs/second, # remain {nremain}, time remain {round(tremain / (60**2), 0)} hours\r", end='')
    


