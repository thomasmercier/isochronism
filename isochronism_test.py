import isochronism
import numpy as np

N = 100
fname = 'temp.txt'
a = np.empty((100, 2))
a[:,0] = np.linspace(-10, 10, 100)
a[:,1] = np.linspace(-10, 10, 100)
np.savetxt(fname, a)


I = 1. / 4. / np.pi**2
bs = isochronism.BalanceSpring(I, fname)
print(bs.frequency(50))
