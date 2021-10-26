from __future__ import print_function
import os, sys
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, merge
from keras.layers import Reshape, Input
from keras.layers.core import Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from PIL import Image
import argparse
import math
from data import load_LIDSet1, mnist_shuffle_data,shuffle_data


n_classes=50
feature_dim=49
feature_dim_g=49
raw_dim      =400


def generator_model():  
    M=7
    model_1 = Sequential()
    #n_classes number
    model_1.add(Dense(input_dim=feature_dim_g, output_dim=512))
    model_1.add(Activation('tanh'))
    #Noise
    model_2 = Sequential()
    model_2.add(Dense(input_dim=100, output_dim=512))
    model_2.add(Activation('tanh'))

    model = Sequential()
    model.add(Merge([model_1,model_2], mode='concat', concat_axis=1))
    model.add(Activation('tanh'))
    model.add(Dense(128*M*M))
    model.add(BatchNormalization(mode=0))
    model.add(Activation('tanh'))
    model.add(Reshape((128, M, M), input_shape=(128*M*M,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(feature_dim))
    model.add(Activation('tanh'))    
    return model
    
def discriminator_model():
    
    M=7
    model_1 = Sequential()
    model_1.add(Dense(input_dim=feature_dim, output_dim=512))
    model_1.add(Activation('tanh'))
    #model_1.add(Dropout(0.3))
    model_2 = Sequential()
    model_2.add(Dense(input_dim=feature_dim, output_dim=512))
    #model_2.add(Dense(input_dim=raw_dim, output_dim=512))

    model_2.add(Activation('tanh'))
    #model_2.add(Dropout(0.3))

    model = Sequential()
    model.add(Merge([model_1,model_2], mode='concat', concat_axis=1))
    model.add(Activation('tanh'))
    # 512, 512 big0
    # 2048, 1024 big1
    # 2048, 2048, 1024 big2
    # 4096, 2048, 1024 big3
    # 4096, 4096, 1024 big4
    # 4096, 2048, 2048, 1024 big5
    # 4096, 4096, 2048, 2048, 1024, 1024, 512 big6
    #model.add(Dense(2048))
    #model.add(Activation('tanh'))     
    model.add(Dense(4096))
    model.add(Activation('tanh'))    
    model.add(Dense(2048))
    model.add(Activation('tanh'))    
    #model.add(Dense(512))
    #model.add(Activation('tanh'))
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    #model.add(Dense(512))
    #model.add(Activation('tanh'))
    #model.add(Dense(512))
    #model.add(Activation('tanh'))
    
    model.add(Dense(n_classes+1))
    model.add(Activation('softmax'))
    return model
    

def discriminator_model_withDimReduct():
    
    M=7
    DimeReduct  =49
    model_1     = Sequential()
    model_1.add(Dense(input_dim=raw_dim, output_dim=DimReduct))
    model_1.add(Activation('tanh'))

    model_1.add(Dense(512))
    model_1.add(Activation('tanh'))
    #model_1.add(Dropout(0.3))

    model_2 = Sequential()
    model_2.add(Dense(input_dim=DimReduct, output_dim=512))
    #model_2.add(Dense(input_dim=raw_dim, output_dim=512))
    model_2.add(Activation('tanh'))
    #model_2.add(Dropout(0.3))

    model = Sequential()
    model.add(Merge([model_1,model_2], mode='concat', concat_axis=1))
    model.add(Activation('tanh'))
    # 512, 512 big0
    # 2048, 1024 big1
    # 2048, 2048, 1024 big2
    # 4096, 2048, 1024 big3
    # 4096, 4096, 1024 big4
    # 4096, 2048, 2048, 1024 big5
    # 4096, 4096, 2048, 2048, 1024, 1024, 512 big6
    #model.add(Dense(2048))
    #model.add(Activation('tanh'))     
    model.add(Dense(4096))
    model.add(Activation('tanh'))    
    model.add(Dense(2048))
    model.add(Activation('tanh'))    
    #model.add(Dense(512))
    #model.add(Activation('tanh'))
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    #model.add(Dense(512))
    #model.add(Activation('tanh'))
    #model.add(Dense(512))
    #model.add(Activation('tanh'))
    
    model.add(Dense(n_classes+1))
    model.add(Activation('softmax'))
    return model


def generator_containing_discriminator(generator, discriminator):
    inputs1 = Input((feature_dim_g,))
    inputs2 = Input((100,))
    inputs3 = Input((feature_dim,))
    x_generator = generator([inputs1,inputs2])
    #merged = merge([inputs1, x_generator], mode='concat',concat_axis=1)
    discriminator.trainable = False
    x_discriminator = discriminator([inputs3, x_generator])
    #model = Model(input=inputs, output=[x_generator,x_discriminator])
    model = Model(input=[inputs1, inputs2, inputs3], output=[x_discriminator])
    return model
    

def classifier_model_00():
    M=7
    model_1 = Sequential()
    model_1.add(Dense(input_dim=feature_dim, output_dim=512))
    model_1.add(Activation('tanh'))
    #model_1.add(Dropout(0.3))
    model_2 = Sequential()
    model_2.add(Dense(input_dim=feature_dim, output_dim=512))
    model_2.add(Activation('tanh'))
    #model_2.add(Dropout(0.3))

    model = Sequential()
    model.add(Merge([model_1,model_2], mode='concat', concat_axis=1))
    #model.add(Dense(input_dim=1568, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*M*M))
    model.add(Activation('tanh'))    
    model.add(Reshape((128, M, M), input_shape=(128*M*M,)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(50))
    model.add(Activation('softmax'))
    return model
    
def classifier_model_0():
    M=7
    model = Sequential()
    model.add(Dense(input_dim=feature_dim, output_dim=512))
    model.add(Activation('tanh'))
    model.add(Dense(128*M*M))
    model.add(Activation('tanh'))    
    model.add(Reshape((128, M, M), input_shape=(128*M*M,)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(50))
    model.add(Activation('softmax'))
    return model      
    
def classifier_model_0_dp():
    M=7
    model = Sequential()
    model.add(Dense(input_dim=feature_dim, output_dim=512))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(128*M*M))
    model.add(Activation('tanh'))    
    model.add(Dropout(0.3))
    model.add(Reshape((128, M, M), input_shape=(128*M*M,)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Activation('softmax'))
    return model   
