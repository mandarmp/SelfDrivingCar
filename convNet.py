import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation



#Image normalization to avoid saturation and make gradients work better.
#    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
#    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
#    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
#    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#    Drop out (0.5)
#    Fully connected: neurons: 100, activation: ELU
#    Fully connected: neurons: 50, activation: ELU
#    Fully connected: neurons: 10, activation: ELU
#    Fully connected: neurons: 1 (output)
#    # the convolution layers are meant to handle feature engineering
#    the fully connected layer for predicting the steering angle.
#    dropout avoids overfitting
#    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 

def build_model(n_rows, n_cols, alpha):
    input_data_shape = (n_rows, n_cols, 1)
    
    #first layer
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_data_shape))
    model.add(Conv2D(24, kernel_size=(5, 5),input_shape = input_data_shape))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(2, 2))
    
    #second layer
    model.add(Conv2D(36, kernel_size=(5, 5)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(2, 2))
    
    #third layer
    model.add(Conv2D(48, kernel_size=(5, 5)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(2, 2))
    
    #fourth layer
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(1, 1))
    
    #fifth layer
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(1, 1))
    
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    #sixth layer
    model.add(Dense(100))
    model.add(Activation('elu'))
    
    #seventh layer
    model.add(Dense(50))
    model.add(Activation('elu'))
    
    #eighth layer
    model.add(Dense(10))
    model.add(Activation('elu'))
    
    #output layer
    model.add(Dense(3))
    
    model.summary()
    
    return model
    
    
    
    