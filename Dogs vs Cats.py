#!/usr/bin/env python
# coding: utf-8

import itertools
import io
import os
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

random_seed = 2
np.random.seed(random_seed)

from tqdm import tqdm
from random import shuffle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam


TRAIN_DIR = 'C:/Users/Kukus/Desktop/Kaggle/Datasets/Dogs vs Cats/train/'
TEST_DIR = 'C:/Users/Kukus/Desktop/Kaggle/Datasets/Dogs vs Cats/test/'
IMG_SIZE = 150
TOTAL_PIXELS = IMG_SIZE * IMG_SIZE
train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]
VCG16 = True
IMAGE_MANIP = False

def label_img(img):
    label = img.split('.')[-3]
    if label == 'cat': return 1
    elif label == 'dog': return 0


def prepare_data(list_of_images):

    x = [] # images as arrays
    y = [] # labels
    
    for image in tqdm(list_of_images):
        img = cv2.resize(cv2.imread(image), (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        x.append(img)
    
    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        #else:
            #print('neither cat nor dog name present in images')
            
    return x, y

X, Y = prepare_data(train_images_dogs_cats)

# Split the train and validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1, random_state=random_seed)

learning_rate = 0.0001
epochs = 25
batch_size = 16
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)

# CNN Model
model = Sequential()

model.add(Convolution2D(64, 3, strides=3, padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, strides=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, strides=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))


#optimizer = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer = Adam()

model.compile(optimizer = optimizer , 
    loss = "binary_crossentropy", 
    metrics=["accuracy"])

model.summary()


# Set a learning rate annealer
#learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
#                                            patience=3, 
#                                            verbose=1, 
#                                            factor=0.5, 
#                                            min_lr=0.00001)

if(IMAGE_MANIP):
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images


#datagen.fit(X_train)

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=10,
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True)


train_generator = datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)
validation_generator = datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)

hist = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    verbose = 1,
    validation_steps=nb_validation_samples // batch_size
)

model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

X_test, Y_test = prepare_data(test_images_dogs_cats)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = datagen.flow(np.array(X_test), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1)

counter = range(1, len(test_images_dogs_cats) + 1)
solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv("solution.csv", index = False)


plt.figure(figsize=(16,4))
plt.subplot(121)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.title(('accuracy = {}\nval_accuracy = {}'.format(round(hist.history['accuracy'][-1],3), 
                                                   round(hist.history['val_accuracy'][-1],3))), fontsize=14)
plt.grid(True)
plt.subplot(122)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.title(('loss = {}\nval_loss = {}'.format(round(hist.history['loss'][-1],3), 
                                           round(hist.history['val_loss'][-1],3))), fontsize=14)
plt.grid(True)
plt.show()

