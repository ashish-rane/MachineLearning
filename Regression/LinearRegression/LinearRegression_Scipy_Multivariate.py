# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:27:32 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

J_hist = pd.DataFrame(columns =['theta', 'cost'])
iterations = 0

house_prices = pd.read_csv('house_prices.csv', header=None, 
                           names=['area', 'noOfBedrooms', 'price'])


# features
# multiple features
X = house_prices[['area', 'noOfBedrooms']]


# observed predictions
y = house_prices['price'].values
y = y.reshape(y.shape[0], 1)

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
initial_theta = np.ones(3).reshape(3,1)

scaler_X = MinMaxScaler()
scaler_X.fit(X)
X_norm = scaler_X.scale(X)


# add X0 feature(all ones)
X_norm = np.column_stack((np.ones(X_norm.shape[0]), X_norm))

# train our hypothesis to minimize cost
res = minimize(costFunc, initial_theta, args=(X_norm, y))

theta = res.x

# Create some test data for which to predict
test_data = {'area':[1200, 1500, 2456, 3575] , 'noOfBedrooms':[1,2,3,4]}
X_test = pd.DataFrame(test_data)

# scale features
X_test = scaler_X.scale(X_test)

# add X0 Feature
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

# Make Predictions
y_pred = hypothesisFunc(theta, X_test)

test_data['price'] = y_pred

result_df = pd.DataFrame(test_data)

'''
# plot the fit
fig = pp.figure()
axes = fig.add_subplot(111, projection='3d')
axes.set_xlabel('area')
axes.set_ylabel('no of bedrooms')
axes.set_zlabel('price')

axes.scatter(house_prices['area'], house_prices['noOfBedrooms'], house_prices['price'], color='r')
axes.plot_trisurf(result_df['area'], result_df['noOfBedrooms'], result_df['price'], color ='blue')
'''