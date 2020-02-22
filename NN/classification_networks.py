# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:04:24 2020

@author: steve
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.models import Model
#from keras.layers import Input, Dense

import numpy as np

from keras.layers import InputLayer

from keras.layers import Conv2D

from keras.layers import Input, Dense



def define_model_1(num_classes=3,shape_input=(32, 32, 1)):


    inputs = Input(shape=shape_input)

    x=Conv2D(32, kernel_size=(3, 3),
                     activation='relu')(inputs)
    x=Conv2D(64, (3, 3), activation='relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    x=Dense(128, activation='relu')(x)
    x=Dropout(0.5)(x)
    predictions=Dense(num_classes, activation='softmax',name='final_output')(x)


    model = Model(inputs=inputs, outputs=predictions)
    print("model is defined")    
    return model

def define_model_2(num_classes=3,shape_input=(32, 32, 1)):


    inputs = Input(shape=shape_input)

    x=Conv2D(16, kernel_size=(3, 3),activation='relu')(inputs)

    x=Conv2D(32, (3, 3), activation='relu')(x)
   
    x=Conv2D(64, (3, 3), activation='relu')(x)
  
    x=Conv2D(128, (3, 3), activation='relu')(x)    
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)    

    x=Flatten()(x)
    x=Dense(128, activation='relu')(x)
    x=Dropout(0.5)(x)
    predictions=Dense(num_classes, activation='softmax',name='final_output')(x)


    model = Model(inputs=inputs, outputs=predictions)
    
    return model

def paper_model(num_classes,shape_input_=(32, 32, 1)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=shape_input_))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    print("model is defined")
    return model


def train_model(model, x_train, y_train,x_test, y_test, batch_size=128,epochs=30):


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print("training the model")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print("done training the model")
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    

"""

the following few just loads mnist dataset for testing purposes.


"""    
 

def give_mnist():
        
    
    #img_rows, img_cols = 28, 28
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()[:10000]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test
    


"""

the following can be changed as see fit

"""

# pipeline 

#(1) load the data    
x_train, y_train, x_test, y_test=give_mnist()


#(2) define the model   

#model=paper_model(10,shape_input_=(28, 28, 1))

model=define_model_1(num_classes=10,shape_input=(28, 28, 1))



#(3) fit the model   

train_model(model, x_train, y_train,x_test, y_test)
















