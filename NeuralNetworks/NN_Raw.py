# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:09:36 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

# load the training data
df_train = pd.read_csv('train.csv')

X = df_train.iloc[:,1:].values
y = df_train.iloc[:,0].values

# Visualizing the data
# select random rows to display 
idx = np.random.randint(low = 0, high= X.shape[0], size=100)

sel = X[idx, :]

def displayData(X):
    width=int(np.round(np.sqrt(X.shape[1])))
    (m,n) = X.shape
    height = int(n/width)
    
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m/display_rows))
    
    padding = 1
    
    fig, axes = pp.subplots( nrows=display_rows, ncols=display_cols, figsize=(20,10))
    pp.subplots_adjust(hspace = 0.01, wspace=0.01)
    k = 0
    for i in range(display_rows):
        for j in range(display_cols):
            axes[i,j].imshow(X[k].reshape(height, width), cmap='gray')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            axes[i,j].set_xticklabels([])
            axes[i,j].set_yticklabels([])
            k = k + 1

#displayData(sel)


#-------------------------------------------------------------------

# helper functon to randomly initializeweights
def getInitialWeights(layer1, layer2):
    #W = np.zeros(layer2, 1 + layer1)    # 1 indicates bias units
    ep = 0.12
    W = np.random.random_sample(size=(layer2, layer1 + 1)) * ((2 * ep) - ep)
    return W


# used to convert output y to vectors using one hot encoding
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1-sigmoid(z))
    
def nnCostFunc(nn_params, layers, X, y, reg_factor):
    # reshape our parameter matries from their flatten versions
    theta1 = nn_params[0: layers[1] * (layers[0] + 1)].reshape((layers[1], layers[0] + 1))
    theta2 = nn_params[layers[1] * (layers[0] + 1):].reshape(layers[2], layers[1] + 1)
    
    m = X.shape[0]
    
    X = np.column_stack((np.ones((m,1)), X))
    
    # convert y from pure number to a one hot vector
    y_matrix = get_one_hot(y, nn_layers[-1])

    # first layer
    a1 = X # ( 42000, 401)
    z2 = a1.dot(theta1.T) # 25 x 401 => 42000 x 25
    
    # 2nd Layer
    a2 = np.column_stack((np.ones((m, 1)), sigmoid(z2))) # 42000 x 26
    z3 = a2.dot(theta2.T) # Theta2 10x26 => z3 42000x10
    
    # 3rd Layer
    a3 = sigmoid(z3)
    
    # Our hypothesis prediction
    h = a3

    # the cost is 1 x 10 vector with probabilities of each class for 
    # ith training example.
    # Our cost function for logistic regression
    costK = (-y_matrix *  np.log(h) ) - ((1- y_matrix) * np.log(1 - h))

    intermediate_cost = np.sum(costK)

    # Back Propogation
    # Error Deltas
    d3 = h - y_matrix
    # ignore the first column from theta2 since it is the bias unit
    d2 = (d3.dot(theta2[:,1:])) * sigmoidGradient(z2) # 42000x10 * 10x25 => 42000x25
    
    # Accumulators of errors
    Delta1 = np.zeros((theta1.shape))
    Delta2 = np.zeros((theta2.shape))
    
    Delta2 = Delta2 + (d3.T.dot(a2)) # 10x42000 * 42000x26 => 10x26
    
    Delta1 = Delta1 + (d2.T.dot(a1)) # 25x42000 * 42000x401 => 25x401
    
    # Unregularized cost
    unreg_cost = intermediate_cost/m
    
    # Unregularized gradients
    theta1_unreg = Delta1/m
    theta2_unreg = Delta2/m
    
    # Regularization
    reg_term = (reg_factor * (np.sum(theta1[:,1:] ** 2) + \
                              np.sum(theta2[:,1:] ** 2))) \
                              / (2 * m)
    
    J = unreg_cost + reg_term
    
    # Regularization term of gradients
    R1 = (reg_factor * theta1)/m;
    R2 = (reg_factor * theta2)/m;

    # Set the bias units as unregularized
    theta1_grad = np.copy(theta1_unreg)
    theta2_grad = np.copy(theta2_unreg)
    
    # Set the remaining by adding the regularization term
    theta1_grad[:, 1:] = theta1_unreg[:, 1:] + R1[:,1:]
    theta2_grad[:, 1:] = theta2_unreg[:, 1:] + R2[:,1:] 
    
    # unroll(flatten) the parameters
    grad = np.concatenate(np.ravel(theta1_grad), np.ravel(theta2_grad))
    
    return (J, grad)
    
        
#=================================================================
# define our neural network
# input layer 784, hidden layer 25, output layer 10
nn_layers = [784, 25, 10]
initial_Theta1 = getInitialWeights(nn_layers[0], nn_layers[1])
initial_Theta2 = getInitialWeights(nn_layers[1], nn_layers[2])

# need to unroll (flatten) parameters
initial_nn_params = np.concatenate((np.ravel(initial_Theta1), np.ravel(initial_Theta2)))

# mimimize our cost
from scipy.optimize import minimize

# regularization factor

res = minimize(nnCostFunc, initial_nn_params, args=(nn_layers, X, y, 1))    
    
theta = res.x

# Visualize the hidden layer of the neural network
displayData(theta[:,1:])    
    
