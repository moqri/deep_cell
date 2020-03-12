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

from sklearn.model_selection import train_test_split


import pickle

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
   
    x=Conv2D(64, (9, 9), activation='relu')(x)
  
    x=Conv2D(128, (17, 17), activation='relu')(x)    
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)    

    x=Flatten()(x)
    x=Dense(128, activation='relu')(x)
    x=Dropout(0.5)(x)
    predictions=Dense(num_classes, activation='softmax',name='final_output')(x)


    model = Model(inputs=inputs, outputs=predictions)
    
    return model

def define_model_3(num_classes,shape_input=(32, 32, 1)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=shape_input))
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


def train_model(model, x_train, y_train,x_test, y_test, batch_size=128,epochs=30,l_r=0.05,beta1=0.9,beta2=0.999):


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=l_r,beta_1=beta1,beta_2=beta2),
                  metrics=['accuracy'])
    print("training the model")
    history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    print("done training the model")
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history
    
    

"""

the following few just loads mnist dataset for testing purposes.


"""    
 


def prepare_data(X,y,test_size=0.05):        
    
    img_rows, img_cols = 150, 150
    
    # the data, split between train and test sets
    x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=42)


    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train 
    x_test 
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)
    
    return x_train, y_train, x_test, y_test   
    


"""

the following can be changed as see fit

"""

# pipeline 

#(1) load the data    

folder='C:/CS230/dendritic'


with open('C:/CS230/X_Data_all.pkl','rb') as f:
    X_ = pickle.load(f)
    print(X_.shape)

with open('C:/CS230/y_Data_all.pkl','rb') as f:
    y_ = pickle.load(f)
    print(y_.shape)




x_train, y_train, x_test, y_test = prepare_data(X_,y_)


#(2) define the model   


model=define_model_3(num_classes=2,shape_input=(150, 150, 1))


# Fit the model



#(3) fit the model   

history=train_model(model, x_train, y_train,x_test, y_test, batch_size=64,epochs=30,l_r=0.005,beta1=0.9,beta2=0.999)



"""
refs: 
https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
"""


"""


model=define_model_1(num_classes=2,shape_input=(150, 150, 1))





history=train_model(model, x_train, y_train,x_test, y_test, batch_size=32,epochs=30,l_r=0.05,beta1=0.9,beta2=0.999)



"""










