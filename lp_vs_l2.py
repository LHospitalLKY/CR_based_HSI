import numpy as np
import random
import matplotlib.pyplot as plt
from CRC_computing import *
from pCRC_Computing import *

# 这个文件用于绘制

d = 100
N = 500

y = np.random.rand(d, 1)
X = np.random.rand(d, N)

pCRC = pCRC_Computing(y, X, 0.001, 1.0)
CRC = CRC_Computing(y, X, 0.001)

print(pCRC.alpha)
print('\n')
print(CRC.alpha)

print(pCRC.residual)
print(CRC.residual)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(pCRC.alpha)
ax1.set_ylim(-0.35, 0.35)
ax2 = fig.add_subplot(212)
ax2.set_ylim(-0.35, 0.35)
ax2.plot(CRC.alpha)

plt.show()


