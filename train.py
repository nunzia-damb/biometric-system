from keras.layers import Input, Dense, InputLayer, Conv2D, MaxPooling2D, UpSampling2D, InputLayer, Concatenate, Flatten, Reshape, Lambda, Embedding, dot
from keras.models import Model, load_model, Sequential
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt,pickle
import keras.backend as K
from sklearn.model_selection import train_test_split
import os, sys, numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model

def train_keystrokes_rec(X1,X2,y):
    
    #text encoder
    input_layer = Input((31,1,1))
    layer1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    layer2 = MaxPooling2D((2, 2), padding='same')(layer1)
    layer3 = Conv2D(8, (3, 3), activation='relu', padding='same')(layer2)
    layer4 = MaxPooling2D((2, 2), padding='same')(layer3)
    layer5 = Flatten()(layer4)
    embeddings = Dense(16, activation=None)(layer5)
    norm_embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    #create model
    model = Model(inputs = input_layer, outputs = norm_embeddings)

    #create siamese model
    input1 = Input((31,1,1))
    input2 = Input((31,1,1))

    #left and right
    left_model = model(input1)
    right_model = model(input2)

    #dot product layer
    dot_product = dot([left_model, right_model], axes=1, normalize=False)
    siamese_model = Model(inputs=[input1, input2], outputs=dot_product)

    #summary
    print(siamese_model.summary())

    #compile model
    siamese_model.compile(optimizer='adam', loss='mse')

    #plot flowchart to model
    plot_model(siamese_model, to_file=os.getcwd()+'/siamese_model_mnist.png', show_shapes=1, show_layer_names=1)

    # Fit model
    siamese_model.fit([X1, X2], y, epochs=10, batch_size=5, shuffle=True, verbose=True)

    model.save(os.getcwd()+"/text_encoder.h5","wb")
    siamese_model.save(os.getcwd()+"/text_siamese_model.h5","wb")


    return model, siamese_model