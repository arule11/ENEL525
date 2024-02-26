# ENEL525 Lab 3 Part 2
# Athena McNeil-Roberts 30042085

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from PIL import Image
from sklearn import preprocessing

def mse(array): 
    return np.mean(array.flatten() ** 2)

def corrCoef(T, A): 
    return scipy.stats.pearsonr(T, A).statistic


# Input value - Load and preprocess all 26 images
img1 = np.asarray(Image.open('Lab3-chars/char_a.bmp')).reshape(400,)
img2 = np.asarray(Image.open('Lab3-chars/char_b.bmp')).reshape(400,)
img3 = np.asarray(Image.open('Lab3-chars/char_c.bmp')).reshape(400,)
img4 = np.asarray(Image.open('Lab3-chars/char_d.bmp')).reshape(400,)
img5 = np.asarray(Image.open('Lab3-chars/char_e.bmp')).reshape(400,)
img6 = np.asarray(Image.open('Lab3-chars/char_f.bmp')).reshape(400,)
img7 = np.asarray(Image.open('Lab3-chars/char_g.bmp')).reshape(400,)
img8 = np.asarray(Image.open('Lab3-chars/char_h.bmp')).reshape(400,)
img9 = np.asarray(Image.open('Lab3-chars/char_i.bmp')).reshape(400,)
img10 = np.asarray(Image.open('Lab3-chars/char_j.bmp')).reshape(400,)
img11 = np.asarray(Image.open('Lab3-chars/char_k.bmp')).reshape(400,)
img12 = np.asarray(Image.open('Lab3-chars/char_l.bmp')).reshape(400,)
img13 = np.asarray(Image.open('Lab3-chars/char_m.bmp')).reshape(400,)
img14 = np.asarray(Image.open('Lab3-chars/char_n.bmp')).reshape(400,)
img15 = np.asarray(Image.open('Lab3-chars/char_o.bmp')).reshape(400,)
img16 = np.asarray(Image.open('Lab3-chars/char_p.bmp')).reshape(400,)
img17 = np.asarray(Image.open('Lab3-chars/char_q.bmp')).reshape(400,)
img18 = np.asarray(Image.open('Lab3-chars/char_r.bmp')).reshape(400,)
img19 = np.asarray(Image.open('Lab3-chars/char_s.bmp')).reshape(400,)
img20 = np.asarray(Image.open('Lab3-chars/char_t.bmp')).reshape(400,)
img21 = np.asarray(Image.open('Lab3-chars/char_u.bmp')).reshape(400,)
img22 = np.asarray(Image.open('Lab3-chars/char_v.bmp')).reshape(400,)
img23 = np.asarray(Image.open('Lab3-chars/char_w.bmp')).reshape(400,)
img24 = np.asarray(Image.open('Lab3-chars/char_x.bmp')).reshape(400,)
img25 = np.asarray(Image.open('Lab3-chars/char_y.bmp')).reshape(400,)
img26 = np.asarray(Image.open('Lab3-chars/char_z.bmp')).reshape(400,)

# Normalize input vectors
P = np.array(preprocessing.normalize([img1,img2,img3,img4,img5,img6,img7,img8,img9,img10,img11,img12,img13,img14,
              img15,img16,img17,img18,img19,img20,img21,img22,img23,img24,img25, img26])).T # (400, 26)
# target array
T = np.array([img1,img2,img3,img4,img5,img6,img7,img8,img9,img10,img11,img12,img13,img14,
              img15,img16,img17,img18,img19,img20,img21,img22,img23,img24,img25, img26]).T # (400, 26)

W = np.zeros((400, 400))
b = np.zeros((400, 1))
# thres = 1e-6
thres = 100
rate = 0.04
errors = [15000]
i = 0
diff = 1

while errors[i] >= thres:
# while diff >= thres:
    err = np.zeros((400, 26))

    for j in range(26):
        err[:, j] = T[:, j] - (W.dot(P[:, j]) + b[:, 0])
        b = b + 2 * rate * np.reshape(err[:, j], (400, 1))
        W = W + 2 * rate * np.outer(err[:, j], P[:, j])

    errors.append(mse(err))
    # diff = errors[i] - errors[i + 1]
    i = i + 1

# learning error curve
plt.semilogy(errors)
plt.title("Learning Error Curve")
plt.xlabel('# of Iterations')
plt.ylabel('Mean Square Error')


errorMeasures = np.zeros((26, 26))
# Apply the trained network to recognize the 26 training characters
# correlation coefficient between the network output and desired target output
A = np.zeros((400, 26))
for j in range(26):
    A[:, j] = W.dot(P[:, j]) + b[:, 0]
    errorMeasures[j, j] = corrCoef(T[:,j], A[:,j])

# table of squared error values between each target vector and the output
table = pd.DataFrame(errorMeasures, columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8','P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16',
                                               'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24','P25', 'P26'], 
                     index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8','T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16',
                                               'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24','T25', 'T26'])
print(table)

# plot original images
fig = plt.figure(figsize=(10, 4))
fig.suptitle("Inputs")
plt.subplots_adjust(hspace=0.6, wspace=0.9)

for i in range(26):
    fig.add_subplot(4, 13, i+1, title=f"{i+1}")
    plt.imshow(np.reshape(T[:, i], (20, 20)) )
    
# plot output images
fig2 = plt.figure(figsize=(10, 6))
fig2.suptitle("Outputs")
for i in range(26):
    fig2.add_subplot(4, 13, i+1, title=f"{i+1}")
    plt.imshow(np.reshape(A[:, i], (20, 20)) )

plt.show()