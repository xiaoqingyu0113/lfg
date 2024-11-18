from lfg.derive import dJ
import numpy as np


v = np.array([1, 1, 3])
w = np.array([4, 2, 6])

J = dJ(v, w)
print(J[:9,:])    