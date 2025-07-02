import matplotlib.pyplot as plt
import numpy as np

E = np.loadtxt('mapSz.dat')[:,0:5].T

#plt.plot(E[0],np.mean(E[1:5], axis=0))
plt.plot(E[0],E[1])
plt.plot(E[0],E[2])

plt.show()