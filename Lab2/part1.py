# ENEL525 Lab 2 Part 1
# Athena McNeil-Roberts

# To perform face recognition using linear associator based on Hebbian learning rule and pseudo inverse rule.

from sklearn import preprocessing
import numpy as np
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy

def hardlim(n):
    if(n >= 0):
        return 1
    else:
        return -1
    
def corrCoef(P, A): 
    return scipy.stats.pearsonr(P, np.reshape(A, (A.size,)))

P1 = np.array([1, -1, -1, -1, 1,  -1, 1, 1, 1, -1,  -1, 1, 1, 1, -1, -1, 1, 1, 1, -1,  -1, 1, 1, 1, -1, 1, -1, -1, -1, 1])
P2 = np.array([1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,])
P3 = np.array([-1, -1, -1, 1, 1,  1, 1, 1, -1, 1,  1, 1, 1, -1, 1,  1, -1, -1, 1, 1,  1, -1, 1, 1, 1,  1, -1, -1, -1, -1])
length = len(P1)

# Plot original input
fig = plt.figure(figsize=(10, 7))
fig.suptitle("Lab 2 Part 1")
plt.subplots_adjust(hspace=0.6, wspace=0.5)
fig.add_subplot(4, 3, 1, title="Pattern 1")
plt.imshow(np.reshape(P1, (6, 5)))
fig.add_subplot(4, 3, 2, title="Pattern 2")
plt.imshow(np.reshape(P2, (6, 5)))
fig.add_subplot(4, 3, 3, title="Pattern 3")
plt.imshow(np.reshape(P3, (6, 5)))

# target array
T = np.array([P1, P2, P3]).T # (30, 3)

# Normalize input vectors
P = np.array(preprocessing.normalize([P1, P2, P3])).T # # (30, 3)

# calcuate weight vector with Hebbian
W = np.dot(T, P.T) # (30, 30)

noisy_P1 = P1
noisy_P2 = P2
noisy_P3 = P3

# Randomly reverse 3 pixel values of each given pattern
for i in range(3):
    index = random.randint(0, length - 1)
    noisy_P1[index] = -1 * noisy_P1[index]
    noisy_P2[index] = -1 * noisy_P2[index]
    noisy_P3[index] = -1 * noisy_P3[index]

normalized_noisyP1 = preprocessing.normalize([noisy_P1])
normalized_noisyP2 = preprocessing.normalize([noisy_P2])
normalized_noisyP3 = preprocessing.normalize([noisy_P3]) # (30, 1)

fig.add_subplot(4, 3, 4, title="Noisy pattern 1")
plt.imshow(np.reshape(noisy_P1, (6, 5)))
fig.add_subplot(4, 3, 5, title="Noisy pattern 2")
plt.imshow(np.reshape(noisy_P2, (6, 5)))
fig.add_subplot(4, 3, 6, title="Noisy pattern 3")
plt.imshow(np.reshape(noisy_P3, (6, 5)))

# Apply the trained network to recognize the noisy patterns
A1 = np.dot(W, normalized_noisyP1.T) 
A2 = np.dot(W, normalized_noisyP2.T) 
A3 = np.dot(W, normalized_noisyP3.T) # (30, 1)

# 🐸 CHECK HARD LIM
# A1 = np.array([hardlim(i) for i in A1])
# A2 = np.array([hardlim(i) for i in A2])
# A3 = np.array([hardlim(i) for i in A3])

# plot Hebbian outputs
fig.add_subplot(4, 3, 7, title="Hebbian Output 1")
plt.imshow(np.reshape(A1, (6, 5)))
fig.add_subplot(4, 3, 8, title="Hebbian Output 2")
plt.imshow(np.reshape(A2, (6, 5)))
fig.add_subplot(4, 3, 9, title="Hebbian Output 3")
plt.imshow(np.reshape(A3, (6, 5)))

# Hebbian correlation coefficients 
corrTable = [
    ['Pattern 1', corrCoef(P1, A1).statistic, corrCoef(P1, A2).statistic, corrCoef(P1, A3).statistic], 
    ['Pattern 2', corrCoef(P2, A1).statistic, corrCoef(P2, A2).statistic, corrCoef(P2, A3).statistic], 
    ['Pattern 3', corrCoef(P3, A1).statistic, corrCoef(P3, A2).statistic, corrCoef(P3, A3).statistic]]

print("Hebbian Correlation Coefficient Table")
print(tabulate(corrTable, headers=['', 'Output 1', 'Output 2', 'Output 3'], tablefmt="mixed_grid"))


#  PSUEDO INVERSE :

# 🐸 CHECK P INPUTS - NORMAILIZED OR NOT
# P = np.array([P1, P2, P3]).T

# psuedo inverse of inputs P
PpsuedoInv = np.linalg.pinv(P) # (3, 30)

# calculate weight vector with psuedo inverse
pi_W = np.dot(T, PpsuedoInv) # (30, 30)

# Apply the trained network to recognize the noisy patterns
pi_A1 = np.dot(pi_W, normalized_noisyP1.T) 
pi_A2 = np.dot(pi_W, normalized_noisyP2.T) 
pi_A3 = np.dot(pi_W, normalized_noisyP3.T)

# plot psuedo inverse outputs
fig.add_subplot(4, 3, 10, title="Psuedo Inverse Output 1")
plt.imshow(np.reshape(pi_A1, (6, 5)))
fig.add_subplot(4, 3, 11, title="Psuedo Inverse Output 2")
plt.imshow(np.reshape(pi_A2, (6, 5)))
fig.add_subplot(4, 3, 12, title="Psuedo Inverse Output 3")
plt.imshow(np.reshape(pi_A3, (6, 5)))

# psuedo inverse correlation coefficients 
piCorrTable = [
    ['Pattern 1', corrCoef(P1, pi_A1).statistic, corrCoef(P1, pi_A2).statistic, corrCoef(P1, pi_A3).statistic], 
    ['Pattern 2', corrCoef(P2, pi_A1).statistic, corrCoef(P2, pi_A2).statistic, corrCoef(P2, pi_A3).statistic], 
    ['Pattern 3', corrCoef(P3, pi_A1).statistic, corrCoef(P3, pi_A2).statistic, corrCoef(P3, pi_A3).statistic]]

print("Psuedo Inverse Correlation Coefficient Table")
print(tabulate(piCorrTable, headers=['', 'Output 1', 'Output 2', 'Output 3'], tablefmt="mixed_grid"))

plt.show()
