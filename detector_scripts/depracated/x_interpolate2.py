#! /usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# create a vertical line of data after a horizontal line of data
xa = np.linspace(1, 2, 5)
xb = np.ones((5,)) * 2
ya = np.ones((5,)) * 4
yb = np.linspace(0, 4, num=5, endpoint=True)

x = np.concatenate((xa, xb))
y = np.concatenate((ya, yb))

print(x)
print(y)
xi = 0.5
fx = interp1d(x, y, bounds_error=False, fill_value=(y[0], 0))
yi = fx(xi)

# find the recall value for max precision:
# indices of max values of y
iprec = np.where(y == np.amax(y))
print(iprec)
# find indices of max values of iprec in x
irec = np.where(x[iprec] == np.amax(x[iprec]))
print(irec)
print('max x for max y: {}'.format(x[iprec]))
max_r_for_max_p = np.amax(x[iprec])

# if input is too large, reset input to max value of input/min value of input?
fy = interp1d(y, x, bounds_error=False, fill_value=(0, 0))
yi2 = 4
xi2 = fy(yi2)

plt.plot(x, y, marker='o')
plt.plot(xi, yi, marker='x', color='red')
plt.plot(xi2, yi2, marker='^', color='pink')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

import code
code.interact(local=dict(globals(), **locals()))

# TODO
# interpolate for confidence threshold
# if P > bounds, then return max thresh
# if R > bounds, then return min thresh
# else, interpolate wrt recall if given, interpolate wrt precision if given
# if P and R given, interpolate wrt recall