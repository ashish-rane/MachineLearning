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
axes.plot(xs,ys, color='blue')
axes.scatter(xs,ys,color='green', s=100, marker='o')
axes.set_xlabel('area')
axes.set_ylabel('price')