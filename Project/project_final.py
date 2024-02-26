# ENEL525 Project
# Athena McNeil-Roberts
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import os
import cv2
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import confusion_matrix
import seaborn as sns


path = 'flowers' # main folder

dataset = [] # list for images
targets = [] # list for targets
# Read data from reespective folders
for folderName in os.listdir(path): # iterate through each flower folder
    folderPath = os.path.join(path, folderName)

    # set labels for corresponding flower
    target = 4
    if(folderName == 'daisy'):
        target = 0
    elif(folderName == 'dandelion'):
        target = 1
    elif(folderName == 'rose'):
        target = 2
    elif(folderName == 'sunflower'):
        target = 3
    
    for filename in os.listdir(folderPath):
        filePath = os.path.join(folderPath, filename)
        img = cv2.imread(filePath)
        img = cv2.resize(img, (320, 240)) # resize to 320x240
        img = img / 255.0 # normalize
        dataset.append(img) # save image
        targets.append(target)

# find indexes for spliting - 70% = training, 15% = validation, 15% = testing
training = int(len(dataset) * 0.7)
validation = int(len(dataset) * 0.15)
testing = int(len(dataset) * 0.15)

# convert to images to numpy array
dataset = np.array(dataset)
targets = np.array(targets)

# display 10 images from the original dataset
plt.figure("Dataset", figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([1])
    plt.yticks([1])
    plt.grid(False)
    plt.imshow(dataset[i])

np.random.seed(seed=0)
permuted_idx = np.random.permutation(len(dataset)) # shuffle the dataset
# set training as first 70% of the shuffled dataset
img_train = dataset[permuted_idx[:training]]
target_train = targets[permuted_idx[:training]]
target_train = tf.one_hot(target_train, 5)
# set validation as 15% of the shuffled dataset
img_val = dataset[permuted_idx[training:training+validation]]
target_val = targets[permuted_idx[training:training+validation]]
target_val = tf.one_hot(target_val, 5)
# set validation as last 15% of the shuffled dataset
img_test = dataset[permuted_idx[training+validation:]]
target_test = targets[permuted_idx[training+validation:]]
target_test_noone = targets[permuted_idx[training+validation:]]
target_test = tf.one_hot(target_test, 5)

# display 10 images from the testing dataset
plt.figure("Image train", figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([1])
    plt.yticks([1])
    plt.grid(False)
    plt.imshow(img_train[i])

# Create model and add convolution and pooling layers
model = keras.Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(240 ,320, 3)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
# Flatten model and apply dense layers and softmax output activation function
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(img_train, target_train, epochs=5, validation_data=(img_val, target_val))
model.summary()


# predictions on dataset
predictions_main = model.predict(img_test)
predicted_main = np.argmax(predictions_main, axis=1)
# compute the confusion matrix
confusion_mx = confusion_matrix(target_test_noone, predicted_main)
print("Confusion Matrix:", confusion_mx)
accuracy = ( np.trace(confusion_mx) / len(target_test_noone) ) * 100
print("Accuracy: ", accuracy, "%")


# Plot model accuracy and loss
plt.figure("Accuracy")
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.figure("Loss")
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Evaluate the model on the (15% of) dataset 
test_loss, test_acc = model.evaluate(img_test, target_test, verbose=2)
print("Testing Accuracy", test_acc)

plt.show()

# Own test cases
path = "my_photos" # folder containing my photos
images = []
targets = []
for fileName in os.listdir(path): # iterate through each image
    target = 4
    if "daisy" in fileName:
        label = 0
    elif "dandelion" in fileName:
        label = 1
    elif "rose" in fileName:
        label = 2
    elif "sunflower" in fileName:
        label = 3

    filePath = os.path.join(path, fileName)
    img = cv2.imread(filePath)
    img = cv2.resize(img, (320, 240)) # Resize images
    img = img / 255.0 # normalize
    images.append(img) 
    targets.append(np.array(label))

# display all images from my dataset
plt.figure("My Images", figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([1])
    plt.yticks([1])
    plt.grid(False)
    plt.imshow(images[i])

my_dataset = np.array(images)
my_targets = np.array(targets)

np.random.seed(seed=0)
permuted_idx = np.random.permutation(len(my_dataset)) # shuffle the dataset
my_dataset = my_dataset[permuted_idx[:]]
my_targets = my_targets[permuted_idx[:]]
targets_onehot = tf.one_hot(my_targets, 5)

# display all images from my dataset shuffled
plt.figure("My Images Shuffled", figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([1])
    plt.yticks([1])
    plt.grid(False)
    plt.imshow(my_dataset[i])

# evaluate the model on my dataset
test_loss, test_acc = model.evaluate(my_dataset, targets_onehot, verbose=2)
print("Test Accuracy on Custom Images:", test_acc)

# predictions on my dataset
predictions = model.predict(my_dataset)
predicted = np.argmax(predictions, axis=1)
flowerNames = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
print("Actual target:", my_targets)
print("Predicted labels:", predicted)
# compute the confusion matrix
confusion_mx = confusion_matrix(my_targets, predicted)
print("Confusion Matrix:", confusion_mx)
accuracy = ( np.trace(confusion_mx) / len(my_targets) ) * 100
print("Accuracy: ", accuracy, "%")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mx, annot=True, fmt='d', cmap='Blues', xticklabels=flowerNames, yticklabels=flowerNames)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()