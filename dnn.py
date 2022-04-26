from tensorflow.keras import Input, layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import pickle, os
from sklearn.metrics import confusion_matrix  


def create_input_layers(xshapes):
    inputs = []
    for xshape in xshapes:
        input_shape_x = (xshape[1], 1)
        input_x = Input(shape = input_shape_x)
    return inputs


def create_convolution_layers(inputs, num_layers = 0):
    convs = []
    for input_x in inputs:
        conv      = layers.Conv1D(32,  5,  activation='relu', input_shape=input_x.get_shape())(input_x)
        conv      = layers.MaxPooling1D(strides=2, padding='valid')(conv)
        #conv      = layers.Dropout(0.25)(conv)
        for i in range(num_layers):
            conv      = layers.Conv1D(32,  5,  activation='relu')(conv)
            conv      = layers.MaxPooling1D(strides=2, padding='valid')(conv)
            #conv      = layers.Dropout(0.25)(conv)
        convs.append(conv)
    return convs



def pSCNN(xshapes, num_conv_layers):
    inputs = create_input_layers(xshapes)
    convs = create_convolution_layers(inputs, num_layers = num_conv_layers)
    if len(convs) >= 2:
        conv_merge = layers.concatenate(convs)
    else:
        conv_merge = convs[0]
    flat      = layers.Flatten()(conv_merge)
    dense     = layers.Dense(100,  activation='relu')(flat)
    dense     = layers.Dropout(0.2)(dense)
    output    = layers.Dense(1, activation='sigmoid')(dense)
    model     = models.Model(inputs= inputs, outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model