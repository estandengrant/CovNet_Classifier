'''
* CONVOLUTION NETWORK CLASSIFICATION RESEARCH *

This project aims to explore / demonstrate the ability of Neural Networks to successfully categorise and identify image data.

The work here is fairly straightforward but is intended to act as a sample of my work in this area.

'Petimages' file (located in the root directory of this file) contains two subdirectories ('Cats' & 'Dogs'), these folders
contain images of cats and dogs and can be found at the following address - https://www.microsoft.com/en-us/download/details.aspx?id=54765

'''

import os
import random
import sys
import tensorflow
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

print(sys.version)  # Python version for project - TensorFlow and its related libraries are sensitive to version adjustments

classifications = ["Dog", "Cat"]    # Class of image

# Script to read and parse image files.
# Displays example of one Dog image & one Cat image. Do not remove break statement.

for pet in classifications:
    file_path = os.path.join("PetImages", pet)
    for image in os.listdir(file_path):
        image_array = cv2.imread(os.path.join(file_path, image), cv2.IMREAD_GRAYSCALE)
        #plt.imshow(image_array, cmap="gray")
        #plt.show()
        break

# Data Handling

# Following code reads image files and associates them with the respective class (Cat|Dog).

data = []
i = 0
def prep_data():
    for pet in classifications:
        file_path = os.path.join("PetImages", pet)
        classification_number = classifications.index(pet)   # We want the network to output a single digit that represents either Dog (0) or Cat (1)
        for image in os.listdir(file_path):
            try:
                image_array = cv2.imread(os.path.join(file_path, image), cv2.IMREAD_GRAYSCALE)  #We convert image to Greyscale to make it easier for the network to understand the input data.
                resized_image_array = cv2.resize(image_array, (75,75))  #Resize array to ensure input data size is consistent across image population.
                data.append([resized_image_array, classification_number])
            except Exception as e:
                i+=1
                pass
    print(f"Images failed to process: {i}")
prep_data()

random.shuffle(data)    # Shuffle data so network does not classify based on previous classifications pattern (i.e. if we sequenced the data as all Cats then all Dogs, the network will detect this pattern)

X = [feature[0] for feature in data]    #Seperate X (input) data
y = [feature[1] for feature in data]    #Seperate y (output) data

X = np.array(X).reshape(-1, 75, 75, 1) #Re-shape input data
X = X / 255.0   #Normalize data for input (ensure all values fall within 0-1 scale as 255 is maximum possible value)
y = np.array(y) #Tensorflow does not like lists!

# MODEL CONSTRUCTION
# We will use a multi-layer neural nework to process our data.
# Given the nature of the task, the output layer will use a sigmoind activation function (mapping from 0-1)
# We have 2 output options so will use binary crossentrophy loss function

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X,y,batch_size=32, validation_split=0.1, epochs=3)

