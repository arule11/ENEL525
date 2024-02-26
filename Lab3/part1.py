# ENEL525 Lab 3 Part 1
# Athena McNeil-Roberts

# Implement the least mean square error (LMS) learning algorithm for a single layer
# network with multiple neurons and the linear transfer function to solve a classification task and a
# character recognition task.

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

def mse(array): 
    return np.mean(array.flatten() ** 2)

# Input value - select target values for each class
P1 = [1, 1]
P2 = [1, 2]
class1 = [0, 0]

P3 = [2, -1]
P4 = [2, 0]
class2 = [0, 1]

P5 = [-1, 2]
P6 = [-2, 1]
class3 = [1, 0]

P7 = [-1, -1]
P8 = [-2, -2]
class4 = [1, 1]

P = np.array([P1, P2, P3, P4, P5, P6, P7, P8]).T
T = np.array([class1, class1, class2, class2, class3, class3, class4, class4]).T

W = np.zeros((2, 2))
b = np.zeros((2, 1))
thres = 1e-6
# thres = 0.25

# determine the learning rate
Hess = P.T.dot(P) # Hessian = P transposed dot P - from notes
eigenMax = scipy.linalg.eigh(Hess)[0].max()
alphaMax = 1/eigenMax

rate = 0.004
errors = [1]
diff = 1
i = 0

# while errors[i] > thres:
while diff >= thres:
    err = np.zeros((2, 8))

    for j in range(8):
        err[:, j] = T[:, j] - (W.dot(P[:, j]) + b[:, 0])
        W = W + 2 * rate * np.dot(err[:, j], P[:, j].T)
        b = b + 2 * rate * err[:, j]
        
    errors.append(mse(err))
    diff = errors[i] - errors[i + 1]
    i = i + 1

# learning error curve
plt.semilogy(errors)
plt.title("Learning Error Curve")
plt.xlabel('# of Iterations')
plt.ylabel('Mean Square Error')

T = np.array([class1, class2, class3, class4]).T
errorMeasures = np.zeros((4, 8))

# Apply the trained network to classify the input vectors
# MSE between each target vector and the output obtained for each input vector
for i in range(8):
    for j in range(4):
        a = W * P[:, i] + b
        errorMeasures[j, i] = mse(T[:,j] - a)

# table of squared error values between each target vector and the output
table = pd.DataFrame(errorMeasures, columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'], index=['Class1', 'Class2', 'Class3', 'Class4'])
print(table)
plt.show()
