# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:20:50 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[ 1. , -0.5], [-0.8,  1.5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

fig = pp.figure()
ax = fig.gca(projection='3d')


# Plot the 3D surface
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.1, linewidth =1, antialiased=True, cmap=cm.seismic)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
cset = ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
cset = ax.contour(X, Y, Z, zdir='x', offset=-3, cmap=cm.viridis, levels =1)
cset = ax.contour(X, Y, Z, zdir='y', offset=4, cmap=cm.viridis, levels=1)

#ax.set_xlim(-40, 40)
#ax.set_ylim(-40, 40)
#ax.set_zlim(-100, 100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

pp.show()