# ENEL525 Lab 4 Part 2
# Athena McNeil-Roberts

# Design a predictor based on a feed forward neural network with multiple layers using the backpropagation learning algorithm.

import numpy as np
import matplotlib.pyplot as plt

def mse(array): 
    return np.mean(array.flatten() ** 2)

def logsig(x):
  return ( 1 / (1 + np.exp(-x)) )


# Training a neural network using the LMS Algorithm:

P = np.load('data1.npy').T # The size of the data sequence is 180 points  1 x 180


# Design a predictor network with 5 neurons in a hidden layer and 1 output neuron
# Use the log sigmoid activation function for the hidden layer neurons, and a linear activation function for the output neuron.

# Implement and apply the backpropagation learning algorithm to train the network weights and biases (use the same architecture as in part 1). 
# Use only the first 170 data points for training. Set the learning rate to 0.05 and the error threshold to 0.00002. 
# After training, plot the learning error curve. Predict the next 10 points using your predictor (points 170 to 180) 
# and create a figure with both the true points and the predicted points.

# Initialize the weight matrices and biases with random values using numpy.random.normal()
W1 = np.random.normal(size=(5, 2))
b1 = np.random.normal(size=(5, 1))
W2 = np.random.normal(size=(1, 5))
b2 = np.random.normal(size=(1, 1))


# Set the learning rate to 0.05 and the error threshold to 0.00002.
rate = 0.05
thres = 0.00002

errors = [0.1]
i = 0

a1 = np.zeros((1,5))
a2 = 0

# At the end of an iteration (one iteration meaning a pass through the training points 1-190), calculate the mean squared error (MSE) using all the errors obtained from the current iteration.
# Use the log sigmoid activation function for the hidden layer neurons, and a linear activation function for the output neuron.
while errors[i] > thres:
    err = np.zeros(170)
    # Use only the first 170 data points for training.
    # For each point to be predicted, use the previous two data points in the sequence to predict the desired data point.
    for j in range(170):
        a0 = np.array([P[:, j + 1], P[:, j]])
        a1 = logsig( ((W1).dot(a0) + b1 ) )
        
        #  forward propagation:
        a2 = (W2.dot(a1) + b2).reshape(1,)
        err[j] = P[:, j + 2][0] - a2[0] # p at j+2, using previous two points 
        
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
        # W1 = W1 - rate * S1 * P[:, j]
        b1 = b1 - rate * S1
        
    # calculate the mean squared error (MSE) using all the errors obtained from the current iteration
    errors.append(mse(err))
    i = i + 1
    

# learning error curve
plt.figure("Figure 1")
plt.semilogy(errors, color='blueviolet')
plt.title("Learning Error Curve")
plt.xlabel('# of Iterations')
plt.ylabel('Mean Square Error')

# Testing the Neural Network:
# Predict the next 10 points using your predictor (points 170 to 180)
predicted = []
for j in range(10):
    j = j + 168 # 168 + 2 = 170
    a0 = np.array([P[:, j + 1], P[:, j]])
    a1 = logsig( ((W1).dot(a0) + b1 ) )
    a2 = (W2.dot(a1) + b2).reshape(1,)
    predicted.append(a2)
    print("| point: ", j+2, " | predicted: ", round(a2[0], 5), " |  True: ", round(P[:,j+2][0],5), " | \n")
  
trueVals = P[:,170:][0]

plt.figure("Figure 2")
plt.plot(trueVals, color='skyblue', label='True Values')
plt.plot(predicted, color='deeppink', label='Predicted Values'); 
plt.title("True and Predicted values")
plt.grid()
plt.legend()
plt.show()
