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
from scipy import interpolate

J_hist = pd.DataFrame(columns =['theta', 'cost'])
iterations = 0


house_prices = pd.read_csv('house_prices.csv', header=None, 
                           names=['area', 'noOfBedrooms', 'price'])


# features
X = house_prices['area'].values
X = X.reshape(X.shape[0], 1)

# observed predictions
y = house_prices['price'].values
y = y.reshape(y.shape[0], 1)


# Hypothesis function - h(x) = X.theta
def hypothesisFunc(theta, X):
    return X.dot(theta)


# Cost Function - Mean Square Error
def costFunc(theta, X, y):
    m = X.shape[0]
    y_pred = hypothesisFunc(theta,X)
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    
    # Calculate mean squared error
    cost = np.sum((y_pred - y) ** 2)/ (2 * m)
    J_hist.loc[len(J_hist)] = [theta, cost]
    
    
    global iterations 
    iterations = iterations + 1
    return cost



# initial theta
initial_theta = np.ones(2).reshape(2,1)


# add X0 feature(all ones)
X = np.column_stack((np.ones(X.shape[0]), X))

# train our hypothesis to minimize cost
res = minimize(costFunc, initial_theta, args=(X, y))

theta = res.x

# Create some test data for which to predict
xs = np.linspace(house_prices['area'].min(), house_prices['area'].max(), 10)
xs = np.column_stack((np.ones(xs.shape[0]), xs))

# ys = prediction for test values of xs
ys = hypothesisFunc(theta, xs)


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
theta0_steps = J_hist['theta'].apply(lambda x : x[0])
theta1_steps = J_hist['theta'].apply(lambda x : x[1])
J_steps = J_hist['cost']

ax1.set_xlabel('theta 0')
ax1.set_ylabel('theta 1')
ax1.set_zlabel('Cost')

ax1.plot(theta0_steps, theta1_steps, J_steps, zdir = 'z')
# Customize the viewing angle so its easier to see teh scatter points lie
# on the plane y=0
ax1.view_init(elev=30, azim=-25)

xl = theta0_steps.min()
xu = theta0_steps.max()
yl = theta1_steps.min()
yu = theta1_steps.max()

# draw contours
theta0_vals = np.linspace(0, 100000, 100)
theta1_vals = np.linspace(0, 150, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

for i in range(0, T0.shape[0]):
    for j in range(0, T0.shape[1]):
         t = np.array([T0[i,j], T1[i,j]]).reshape(2,)
         J_vals[i,j] = costFunc(t, X, y)


#ax1.plot_wireframe(T1, T2, Z, rstride=1, cstride=1, colors='g')

ax2 = fig.add_subplot(1,2,2)
cs = ax2.contour(T0, T1, J_vals, 15,  cmap= cm.hsv, origin = 'lower', extent=[0, 90000, 0, 150])
ax2.scatter(theta0_steps, theta1_steps, marker='x', color='r')
#ax2.imshow(J_vals, , cmap = cm.jet, alpha=0.9)
ax2.clabel(cs, inline=True, fontsize=10)
#ax2.clim(0,10)
ax2.set_xlabel('theta 0')
ax2.set_ylabel('theta 1')
fig.colorbar(cs, ax=ax2)


########### Using Scaling ############

class MinMaxScaler:
    def __init__(self):
        self.mean = 0
        self.variance = 0
        
    def fit(self, f):
        self.mean = np.mean(f)
        self.variance = np.max(f) - np.min(f)
    
    # f is an numpy array or a series
    def scale(self, f):
        return (f - self.mean)/self.variance
    
    def inverseScale(self, f):
        return f * self.variance + self.mean
    

# clear J_hist
J_hist = J_hist.iloc[0:0]


# features
X = house_prices['area'].values
X = X.reshape(X.shape[0], 1)

# observed predictions
y = house_prices['price'].values
y = y.reshape(y.shape[0], 1)


scaler_X = MinMaxScaler()
scaler_X.fit(X)    
X = scaler_X.scale(X)

scaler_Y = MinMaxScaler()
scaler_Y.fit(y)
y = scaler_Y.scale(y)
y = y.reshape(y.shape[0], 1)

# add X0 feature(all ones)
X = np.column_stack((np.ones(X.shape[0]), X))


# train our hypothesis to minimize cost
res = minimize(costFunc, initial_theta, args=(X, y))

theta = res.x

# Create some test data for which to predict
xs = np.linspace(house_prices['area'].min(), house_prices['area'].max(), 10)
xs = scaler_X.scale(xs)
xs = np.column_stack((np.ones(xs.shape[0]), xs))

ys = hypothesisFunc(theta, xs)


# inverse scale everything
X = scaler_X.inverseScale(X[:,1])
xs = scaler_X.inverseScale(xs[:, 1])

y = scaler_Y.inverseScale(y)
ys = scaler_Y.inverseScale(ys)

fig1, axes = pp.subplots()
axes
axes.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)
axes.scatter(X, y, marker='o', color='r')

axes.plot(xs,ys, color='blue')
axes.set_xlabel('area')
axes.set_ylabel('price')


# Plot Progress
'''
fig = pp.figure()
ax = fig.gca()
theta0_steps = J_hist['theta'].apply(lambda x : x[0])
theta1_steps = J_hist['theta'].apply(lambda x : x[1])
J_steps = J_hist['cost']

xl = theta0_steps.min()
xu = theta0_steps.max()
yl = theta1_steps.min()
yu = theta1_steps.max()

# draw contours
X_scale = scaler_X.scale(X)
X_scale = np.column_stack((np.ones(X_scale.shape[0]), X_scale))
y_scale = scaler_Y.scale(y)
y_scale = y_scale.reshape(y_scale.shape[0], 1)

theta0_vals = np.linspace(2 *xl, 2 * xu, 100)
theta1_vals = np.linspace(2 *yl, 2 *yu, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

for i in range(0, T0.shape[0]):
    for j in range(0, T0.shape[1]):
         t = np.array([T0[i,j], T1[i,j]]).reshape(2,)
         J_vals[i,j] = costFunc(t, X_scale, y_scale)

cs = ax.contour(T0, T1, J_vals, 15,  cmap= cm.hsv, origin = 'lower')
ax.scatter(theta0_steps, theta1_steps, marker='x', color='r', s=100)
ax.clabel(cs, inline=True, fontsize=10)
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
fig.colorbar(cs, ax=ax)
'''