from math import floor, log10
import numpy as np

def signif(x, digits=4):
  return np.round(x, digits-int(floor(log10(abs(x))))-1)


