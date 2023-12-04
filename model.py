#importing the necessary libraries
import tensorflow as tf
from tensorfow import keras
from tensorflow.keras import sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import cv2
import os
import numpy as np
import random

#getting the train and test dataset and initialising the labels
DIRECTORY = r'data/train_test'
CATEGORIES = ['Bears','Pandas']

#merging the labels with the images and resizing them
data = []
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY,category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(128,128))

        data.append([img_arr,label])

#randomly shuffling the data to improve the training of the model
random.shuffle(data)

#creating two empty lists and storing features and labels seperately
x=[]
y=[]
for features,label in data:
    x.append(features)
    y.append(label)

#changing the images into numpy arrays
x = np.array(x)
y = np.array(y)

x = x/225

x.shape

#CNN model
model=Sequential

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,input_shape = x.shape[1:],activation = 'relu'))

model.add(Dense(2,activation = 'softmax'))

#compiling and training the model
model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics=['acuracy'])
model.fit(x,y,epochs=5,validation_split=0.1)

model.save("bear_vs_panda1.h5")
