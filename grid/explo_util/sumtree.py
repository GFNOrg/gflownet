import numpy as np

class SumTree:

  def __init__(self, size):
    self.rng = np.random
    self.nlevels = int(np.ceil(np.log(size) / np.log(2))) + 1
    self.size = size
    self.levels = []
    for i in range(self.nlevels):
      self.levels.append(np.zeros(min(2**i, size), dtype="float32"))

  def sample(self, q=None):
    q = self.rng.random() if q is None else q
    q *= self.levels[0][0]
    s = 0
    for i in range(1, self.nlevels):
      s *= 2
      if self.levels[i][s] < q and self.levels[i][s + 1] > 0:
        q -= self.levels[i][s]
        s += 1
    return s

  def set(self, idx, p):
    delta = p - self.levels[-1][idx]
    for i in range(self.nlevels - 1, -1, -1):
      self.levels[i][idx] += delta
      idx //= 2

  def get(self, idx):
      return self.levels[-1][idx]

  def total(self):
      return self.levels[0][0]
