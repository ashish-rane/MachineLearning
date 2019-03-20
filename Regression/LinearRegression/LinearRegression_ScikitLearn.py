# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:10:19 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

house_prices = pd.read_csv('house_prices.csv', header=None, 
                           names=['area', 'noOfBedrooms', 'price'])

house_prices = house_prices[['area','price']]

house_prices.plot.scatter(x='area', y = 'price', c='r')

X = house_prices['area']
y = house_prices['price']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X.values.reshape((X.shape[0], 1)), y)

# Do predictions
# Create some test data for which to predict
xs = np.linspace(house_prices['area'].min(), house_prices['area'].max(), 10)
xs = xs.reshape((xs.shape[0], 1))

# ys = prediction for test values of xs
ys = linreg.predict(xs)


fig, axes = pp.subplots()
axes.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)
axes.scatter(X, y, marker='o', color='r')

###################################################################3

house_prices = pd.read_csv('house_prices.csv', header=None, 
                           names=['area', 'noOfBedrooms', 'price'])

X = house_prices.iloc[:,:-1 ]
y = house_prices['price']

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split for model evaluation
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=101)

# Train our model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Get Predictions
y_pred = linreg.predict(X_test)

# Evaluation 
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
print ('Mean Absolute Error', mean_absolute_error(y_test, y_pred))
print ('Mean Squared Error', mean_squared_error(y_test, y_pred))
print ('Root Mean Square Error', np.sqrt(mean_absolute_error(y_test, y_pred)))

print ('R-squared', r2_score(y_test,y_pred))