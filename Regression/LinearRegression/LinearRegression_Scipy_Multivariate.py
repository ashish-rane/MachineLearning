# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:27:32 2019

@author: ashish
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

J_hist = pd.DataFrame(columns =['theta', 'cost'])
iterations = 0

def scaleFeatures(X, min_val, std_dev):
    X_norm = (X - min_val) / std_dev
    return X_norm

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
initial_theta = np.ones(3).reshape(3,1)

# multiple features
X = house_prices[['area', 'noOfBedrooms']]

min_val = np.min(X, axis = 0)
std_dev = np.std(X, axis = 0)

X_norm = scaleFeatures(X, min_val, std_dev)

# observed predictions
y = house_prices['price']

# add X0 feature(all ones)
X_norm = np.column_stack((np.ones(X_norm.shape[0]), X_norm))

# train our hypothesis to minimize cost
res = minimize(costFunc, initial_theta, args=(X_norm, y))

theta = res.x

# Create some test data for which to predict
test_data = {'area':[2456, 3575] , 'noOfBedrooms':[3,4]}
X_test = pd.DataFrame(test_data)

# scale features
X_test = scaleFeatures(X_test, min_val, std_dev)

# add X0 Feature
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

# Make Predictions
y_pred = X_test.dot(theta)

test_data['price'] = y_pred

result_df = pd.DataFrame(test_data)
