import numpy as np
import matplotlib.pyplot

matplotlib.pyplot.figure(); matplotlib.pyplot.hold(True)
r = 1
linestyle = ['b-','k-','m-','r-','y-']
p_values = (0.25, 0.5, 1, 2, 4)
for i,p in enumerate(p_values):
    x = np.arange(-r,r+1e-5,1/128.0)
    y = (r**p - (abs(x)**p))**(1.0/p)
    y = zip(y, -y)
    matplotlib.pyplot.plot(x, y, linestyle[i], label='p =' + str(i))
matplotlib.pyplot.axis('equal')
matplotlib.pyplot.legend(loc='upper right')
matplotlib.pyplot.show()
