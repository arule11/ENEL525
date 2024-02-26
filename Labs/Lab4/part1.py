# ENEL525 Lab 4 Part 1
# Athena McNeil-Roberts

# Design a predictor based on a feed forward neural network with multiple layers using the backpropagation learning algorithm.

from sklearn import preprocessing
import numpy as np
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy
import math

def mse(array): 
    return np.mean(array.flatten() ** 2)

def logsig(x):
  return ( 1 / (1 + np.exp(-x)) )


# Implement and apply the backpropagation algorithm to find the weights and biases of 
# the predictor for this chaotic data sequence (use the first 190 data points to train).
# Design a predictor network with 5 neurons in a hidden layer and 1 output neuron
# Use the log sigmoid activation function for the hidden layer neurons, and a linear activation function for the output neuron.


# Training a neural network using the LMS Algorithm:

P = np.zeros((1, 200))
P[0, 0] = 0.35
n = 1

# Generate a data sequence of 200 points using the following chaotic dynamic system: p[n] = 4p[n-1](1-p[n-1]) , with p[0]=0.35.
while n < 200:
    P[0, n] = 4 * P[0, n-1] * (1 - P[0, n-1])
    n = n + 1

# Initialize the weight matrices and biases with random values using numpy.random.normal()
W1 = np.random.normal(size=(5, 2))
b1 = np.random.normal(size= (5, 1))
W2 = np.random.normal(size=(1, 5))
b2 = np.random.normal(size=(1, 1))


# Set the learning rate to 0.1 and the error threshold to 0.015.
rate = 0.1
thres = 0.015

errors = [0.25]
i = 0

a1 = np.zeros((1,5))
a2 = 0

# At the end of an iteration (one iteration meaning a pass through the training points 1-190), calculate the mean squared error (MSE) using all the errors obtained from the current iteration.
# Use the log sigmoid activation function for the hidden layer neurons, and a linear activation function for the output neuron.
while errors[i] > thres:
    err = np.zeros(190)
    for j in range(190):
        # For each point to be predicted, use the previous two data points in the sequence to predict the desired data point.
        a0 = np.array([P[:, j + 1], P[:, j]])
        a1 = logsig( ((W1).dot(a0) + b1 ) )
        
        #  forward propagation:
        # a1 = logsig((W1 * P[:, j]) + b1)
        a2 = (W2.dot(a1) + b2).reshape(1,)
        err[j] = P[:, j + 2][0] - a2[0]
        
        # F matrix based on textbook
        F1 = [ (1 - a1[0, 0])*a1[0, 0], (1 - a1[1, 0])*a1[1, 0], (1 - a1[2, 0])*a1[2, 0], (1 - a1[3, 0])*a1[3, 0], (1 - a1[4, 0])*a1[4, 0]]
        F1 = np.diag(F1)
        F2 = 1
        
        # backpropagate the sensitivies:
        S2 = -2 * F2 * err[j]
        S1 = F1.dot((W2).T) * S2
        
        # update the weights and bias:
        W2 = W2 - rate * S2 * a1.T
        b2 = b2 - rate * S2
        W1 = W1 - rate * (S1.dot(a0.T))
        b1 = b1 - rate * S1
        
    # calculate the mean squared error (MSE) using all the errors obtained from the current iteration
    errors.append(mse(err))
    # print("🦄", mse(err) )
    i = i + 1
    

# learning error curve
plt.figure("Figure 1")
plt.semilogy(errors, color='blueviolet')
plt.title("Learning Error Curve")
plt.xlabel('# of Iterations')
plt.ylabel('Mean Square Error')


# Testing the Neural Network:
predicted = []
for j in range(10):
    j = j + 188
    a0 = np.array([P[:, j + 1], P[:, j]])
    a1 = logsig( ((W1).dot(a0) + b1 ) )
    a2 = (W2.dot(a1) + b2).reshape(1,)
    predicted.append(a2)
    print("\npoint: ", j+2, "  predicted: ", a2, "  True: ", P[:,j+2][0])
  
trueVals = P[:,190:][0]

plt.figure("Figure 2")
plt.plot(trueVals, color='skyblue', label='True Values')
plt.plot(predicted, color='deeppink', label='Predicted Values'); 
plt.title("True and Predicted values")
plt.grid()
plt.legend()
plt.show()
