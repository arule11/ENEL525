# ENEL525 Lab 1
# Athena McNeil-Roberts

import numpy as np
import matplotlib.pyplot as plt

def hardlim(n):
    if(n >= 0):
        return 1
    else:
        return 0

# W = np.array([1, -0.8])
# b = 0
# P = np.array([[1, -1, 0], [2, 2, -1]])
# t = np.array([1, 0, 0])

W = np.array([0,0])
b = 0
P = np.array([[1, 2, 3, 1, 2, 4], [4, 5, 3.5, 0.5, 2, 0.5]])
t = np.array([1, 1, 1, 0, 0, 0])

p1 = np.array([P[0,0], P[1,0]])

length = len(P[0])

error = [1]*length
while any(error):    
    n = np.dot(W, p1) + b
    a = hardlim(n)
    
    for i in range(length):
        pi = P[:,i]
        n = np.dot(W, pi) + b
        a = hardlim(n)
        e = t[i] - a # calculate the error
        if(e != 0): # if error than need to update W & b
            error[i] = 1
            W = W + e*pi.T  # transpose P to do proper matrix multiplication
            b = b + e
        else:
            error[i] = 0
    
print("Weight vector W =", W)
print("Bias b =", b)


# Plot the points on the graph
c1x, c1y, c2x, c2y  = [], [], [], []
for i in range(length):
    if t[i] == 0: # classify points based on target values, t
        c1x.append(P[0,i])
        c1y.append(P[1,i])
    else:  
        c2x.append(P[0,i])
        c2y.append(P[1,i])
plt.plot(c1x, c1y, 'o', color='dodgerblue', label='Class 1')
plt.plot(c2x, c2y, '*', color='deeppink', label='Class 2')


# Plot the decision boundary 
x = np.linspace(1, 4)
m = (-W[0])/(W[1]); # slope of the line
y = m*x + (-b)/(W[1]) # equation of the line
# plt.axline((0, (-b)/(W[1])), slope=m, color='blueviolet', label='Decision Boundary')
plt.plot(x, y, color='blueviolet', label='Decision Boundary'); 

plt.title("Decision Boundary Plot")
plt.grid()
plt.legend()
plt.show()
