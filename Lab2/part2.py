# ENEL525 Lab 2 Part 2
# Athena McNeil-Roberts 30042085

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tabulate import tabulate
import scipy
from skimage.color import rgb2gray
from PIL import Image

def awgn(signal, snr): 
    db_signal = 10 * np.log10(np.mean(signal ** 2)) 
    db_noise = db_signal - snr 
    noise_power = 10 ** (db_noise / 10) 
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal)) 
    return signal + noise

def corrCoef(P, A): 
    return scipy.stats.pearsonr(P, np.reshape(A, (A.size,)))


img1 = Image.open('beyonce.jpg').convert('L')
img2 = Image.open('einstein.jpg').convert('L')
img3 = Image.open('marie-curie.jpg').convert('L')
img4 = Image.open('michael-jackson.jpg').convert('L')
img5 = Image.open('queen.jpg').convert('L')

# create array of numbers from image imputs
P1 = np.asarray(img1)
P2 = np.asarray(img2)
P3 = np.asarray(img3)
P4 = np.asarray(img4)
P5 = np.asarray(img5)

P1 = np.reshape(P1, (P1.size))
P2 = np.reshape(P2, (P2.size))
P3 = np.reshape(P3, (P3.size))
P4 = np.reshape(P4, (P4.size))
P5 = np.reshape(P5, (P5.size)) # (4096, )

# plot original images
fig = plt.figure(figsize=(14, 7))
fig.suptitle("Lab 2 Part 2")
plt.subplots_adjust(hspace=0.6, wspace=0.9)

fig.add_subplot(4, 5, 1, title="Image 1")
plt.imshow(np.reshape(P1, (64, 64)))
fig.add_subplot(4, 5, 2, title="Image 2")
plt.imshow(np.reshape(P2, (64, 64)))
fig.add_subplot(4, 5, 3, title="Image 3")
plt.imshow(np.reshape(P3, (64, 64)))
fig.add_subplot(4, 5, 4, title="Image 4")
plt.imshow(np.reshape(P4, (64, 64)))
fig.add_subplot(4, 5, 5, title="Image 5")
plt.imshow(np.reshape(P5, (64, 64)))

# Normalize input vectors
P = np.array(preprocessing.normalize([P1, P2, P3, P4, P5])).T # (4096, 5)
# P = np.array([normalized_P1[0], normalized_P2[0], normalized_P3[0], normalized_P4[0], normalized_P5[0]]).T # (4096, 5)

# target array
T = np.array([P1, P2, P3, P4, P5]).T # (4096, 5)

# calculate weight vector with Hebbian
W = np.dot(T, P.T) # (4096, 4096)

# add white Gaussian noise with a target SNR of 20 dB to each image
noisy_P1 = awgn(P1, 20)
noisy_P2 = awgn(P2, 20)
noisy_P3 = awgn(P3, 20)
noisy_P4 = awgn(P4, 20)
noisy_P5 = awgn(P5, 20)

# print("🐯", P1[35], noisy_P1[35])
fig.add_subplot(4, 5, 6, title="Noisy Image 1")
plt.imshow(np.reshape(noisy_P1, (64, 64)))
fig.add_subplot(4, 5, 7, title="Noisy Image 2")
plt.imshow(np.reshape(noisy_P2, (64, 64)))
fig.add_subplot(4, 5, 8, title="Noisy Image 3")
plt.imshow(np.reshape(noisy_P3, (64, 64)))
fig.add_subplot(4, 5, 9, title="Noisy Image 4")
plt.imshow(np.reshape(noisy_P4, (64, 64)))
fig.add_subplot(4, 5, 10, title="Noisy Image 5")
plt.imshow(np.reshape(noisy_P5, (64, 64)))

normalized_noisyP1 = preprocessing.normalize([noisy_P1]) # (1, 4096)
normalized_noisyP2 = preprocessing.normalize([noisy_P2])
normalized_noisyP3 = preprocessing.normalize([noisy_P3])
normalized_noisyP4 = preprocessing.normalize([noisy_P4])
normalized_noisyP5 = preprocessing.normalize([noisy_P5])


# Apply the trained network to recognize the noisy patterns
A1 = np.dot(W, normalized_noisyP1.T) # (4096, 1)
A2 = np.dot(W, normalized_noisyP2.T) 
A3 = np.dot(W, normalized_noisyP3.T)
A4 = np.dot(W, normalized_noisyP4.T) 
A5 = np.dot(W, normalized_noisyP5.T)

# plot Hebbian outputs
fig.add_subplot(4, 5, 11, title="Hebbian Output 1")
plt.imshow(np.reshape(A1, (64, 64)))
fig.add_subplot(4, 5, 12, title="Hebbian Output 2")
plt.imshow(np.reshape(A2, (64, 64)))
fig.add_subplot(4, 5, 13, title="Hebbian Output 3")
plt.imshow(np.reshape(A3, (64, 64)))
fig.add_subplot(4, 5, 14, title="Hebbian Output 4")
plt.imshow(np.reshape(A4, (64, 64)))
fig.add_subplot(4, 5, 15, title="Hebbian Output 5")
plt.imshow(np.reshape(A5, (64, 64)))

# Hebbian correlation coefficients 
corrTable = [
    ['Image 1', corrCoef(P1, A1).statistic, corrCoef(P1, A2).statistic, corrCoef(P1, A3).statistic, corrCoef(P1, A4).statistic, corrCoef(P1, A5).statistic], 
    ['Image 2', corrCoef(P2, A1).statistic, corrCoef(P2, A2).statistic, corrCoef(P2, A3).statistic, corrCoef(P2, A4).statistic, corrCoef(P2, A5).statistic], 
    ['Image 3', corrCoef(P3, A1).statistic, corrCoef(P3, A2).statistic, corrCoef(P3, A3).statistic, corrCoef(P3, A4).statistic, corrCoef(P3, A5).statistic],
    ['Image 4', corrCoef(P4, A1).statistic, corrCoef(P4, A2).statistic, corrCoef(P4, A3).statistic, corrCoef(P4, A4).statistic, corrCoef(P4, A5).statistic],
    ['Image 5', corrCoef(P5, A1).statistic, corrCoef(P5, A2).statistic, corrCoef(P5, A3).statistic, corrCoef(P5, A4).statistic, corrCoef(P5, A5).statistic]]

print("Hebbian Correlation Coefficient Table")
print(tabulate(corrTable, headers=['', 'Output 1', 'Output 2', 'Output 3', 'Output 4', 'Output 5'], tablefmt="mixed_grid"))


#  PSUEDO INVERSE :

# 🐸 CHECK P INPUTS - NORMAILIZED OR NOT
# P = np.array([P1, P2, P3, P4, P5]).T
# psuedo inverse of inputs P
PpsuedoInv = np.linalg.pinv(P) # (5, 4096)

# calculate weight vector with psuedo inverse
pi_W = np.dot(T, PpsuedoInv) # (4096, 4096)

# Apply the trained network to recognize the noisy patterns
pi_A1 = np.dot(pi_W, normalized_noisyP1.T) 
pi_A2 = np.dot(pi_W, normalized_noisyP2.T) 
pi_A3 = np.dot(pi_W, normalized_noisyP3.T)
pi_A4 = np.dot(pi_W, normalized_noisyP4.T) 
pi_A5 = np.dot(pi_W, normalized_noisyP5.T)

# plot psuedo inverse outputs
fig.add_subplot(4, 5, 16, title="Psuedo Inverse Output 1")
plt.imshow(np.reshape(pi_A1, (64, 64)))
fig.add_subplot(4, 5, 17, title="Psuedo Inverse Output 2")
plt.imshow(np.reshape(pi_A2, (64, 64)))
fig.add_subplot(4, 5, 18, title="Psuedo Inverse Output 3")
plt.imshow(np.reshape(pi_A3, (64, 64)))
fig.add_subplot(4, 5, 19, title="Psuedo Inverse Output 4")
plt.imshow(np.reshape(pi_A4, (64, 64)))
fig.add_subplot(4, 5, 20, title="Psuedo Inverse Output 5")
plt.imshow(np.reshape(pi_A5, (64, 64)))

# psuedo inverse correlation coefficients 
piCorrTable = [
    ['Image 1', corrCoef(P1, pi_A1).statistic, corrCoef(P1, pi_A2).statistic, corrCoef(P1, pi_A3).statistic, corrCoef(P1, pi_A4).statistic, corrCoef(P1, pi_A5).statistic], 
    ['Image 2', corrCoef(P2, pi_A1).statistic, corrCoef(P2, pi_A2).statistic, corrCoef(P2, pi_A3).statistic, corrCoef(P2, pi_A4).statistic, corrCoef(P2, pi_A5).statistic], 
    ['Image 3', corrCoef(P3, pi_A1).statistic, corrCoef(P3, pi_A2).statistic, corrCoef(P3, pi_A3).statistic, corrCoef(P3, pi_A4).statistic, corrCoef(P3, pi_A5).statistic],
    ['Image 4', corrCoef(P4, pi_A1).statistic, corrCoef(P4, pi_A2).statistic, corrCoef(P4, pi_A3).statistic, corrCoef(P4, pi_A4).statistic, corrCoef(P4, pi_A5).statistic],
    ['Image 5', corrCoef(P5, pi_A1).statistic, corrCoef(P5, pi_A2).statistic, corrCoef(P5, pi_A3).statistic, corrCoef(P5, pi_A4).statistic, corrCoef(P5, pi_A5).statistic]]

print("Psuedo Inverse Correlation Coefficient Table")
print(tabulate(piCorrTable, headers=['', 'Output 1', 'Output 2', 'Output 3', 'Output 4', 'Output 5'], tablefmt="mixed_grid"))

plt.show()

