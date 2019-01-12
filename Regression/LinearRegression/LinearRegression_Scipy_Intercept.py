# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:27:32 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

J_hist = pd.DataFrame(columns =['theta', 'cost'])
iterations = 0

def costFunc(theta, X, y):
    #h(x) = theta0* X0 + theta1*X1
    #print('X:{0}, theta: {1}, y:{2}'.format(X.shape, theta.shape, y.shape))
    y_pred = X.dot(theta)
    # Calculate mean squared error
    mse = ((y_pred - y) ** 2).mean()
    J_hist.loc[len(J_hist)] = [theta, mse]
    global iterations 
    iterations = iterations + 1
    return mse



house_prices = pd.read_csv('house_prices.csv', header=None, 
                           names=['area', 'noOfBedrooms', 'price'])

# initial theta
initial_theta = np.ones(2).reshape(2,1)

# features
X = house_prices['area']

# observed predictions
y = house_prices['price']

# add X0 feature(all ones)
X = np.column_stack((np.ones(X.shape[0]), X))

# train our hypothesis to minimize cost
res = minimize(costFunc, initial_theta, args=(X, y))

theta = res.x

# Create some test data for which to predict
xs = np.linspace(house_prices['area'].min(), house_prices['area'].max(), 10)
xs = np.column_stack((np.ones(xs.shape[0]), xs))

# ys = prediction for test values of xs
ys = xs.dot(theta) 


fig1, axes = pp.subplots()
axes
axes.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)
axes.scatter(X[:, 1], y, marker='o', color='r')

axes.plot(xs[:, 1],ys, color='blue')
axes.set_xlabel('area')
axes.set_ylabel('price')


# Plot Cost Function as function of theta


fig = pp.figure(figsize=(15,4))
ax1 = fig.add_subplot(1,2,1,projection='3d')
x = J_hist['theta'].apply(lambda x : x[0])
y = J_hist['theta'].apply(lambda x : x[1])
z = J_hist['cost']

ax1.set_xlabel('theta 0')
ax1.set_ylabel('theta 1')
ax1.set_zlabel('Cost')

ax1.plot(x, y, z, zdir = 'z')
# Customize the viewing angle so its easier to see teh scatter points lie
# on the plane y=0
ax1.view_init(elev=30, azim=-25)

'''
t1 = np.arange(J_hist['theta'][0].min(), J_hist['theta'][0].min(), 0.5)
t2 = np.arange(J_hist['theta'][1].min(), J_hist['theta'][1].min(), 0.5)
T1, T2 = np.meshgrid(t1,t2)
Z = np.sqrt(  (X**2)/ 2 + Y **2)


#ax1.plot_wireframe(T1, T2, Z, rstride=1, cstride=1, colors='g')

ax2 = fig.add_subplot(1,2,2)
cs = ax2.contour(T1, T2, Z, cmap=cm.hsv, vmin=abs(Z).min(), vmax=abs(Z).max())
fig.colorbar(cs, ax=ax2)


'''