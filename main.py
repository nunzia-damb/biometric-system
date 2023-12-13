import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil
from random import random, randint, seed
import random
import pickle, itertools, sklearn, pandas as pd, seaborn as sn
from scipy.spatial import distance
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.utils import plot_model
from scipy import spatial
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Dense, InputLayer, Conv2D, MaxPooling2D, UpSampling2D, InputLayer, Concatenate, Flatten, Reshape, Lambda, Embedding, dot
from keras.models import Model, load_model, Sequential
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt,pickle
import keras.backend as K
from sklearn.model_selection import train_test_split
import os, sys, numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model

import warnings

warnings.filterwarnings('ignore')

#import model
from train import train_keystrokes_rec

# parse excel
data_frame = pd.read_csv('data/DSL-StrongPasswordData.csv')

# dictionary of subject: list of features for session [ [feature session x] ]
data_by_part = {}

# dictionary with participants as keys and list of metrics as values
# can be used as positive samples
key = ''
columns = list(data_frame.columns)[3:]  # discard useless columns
for i in data_frame.index:
    row = []
    for tag in columns:
        row.append(data_frame[tag][i])
    if data_frame['subject'][i] == key:
        data_by_part[key].append(row)
    else:
        key = data_frame['subject'][i]
        data_by_part[key] = [row]

#Test written
test_data = [data_by_part[key][200:] for key in data_by_part]

# takes as data for training first 20 list of[features session x] for every subject
positive_couples = []
for key in data_by_part:
    # combines 2 features vectors for every user
    positive_couples += list(itertools.combinations(data_by_part[key][:20], 2))

# combines features vectors for different users
negative_couples = []
for key1 in data_by_part:
    for key2 in data_by_part:
        if key2 != key1:
            negative_couples += list(itertools.product(data_by_part[key1][:20], data_by_part[key2][:20]))

text_X1 = []
text_X2 = []
text_y = []
'''
# Create pairs of keystroke features and set target label for them. Target output is 1 if a pair comes from the same
# user else it is 0
text_X1.extend([[sample[0] for sample in positive_couples]])
text_X2.extend([[sample[1] for sample in positive_couples]])
text_y.extend([1] * len(positive_couples))

text_X1.append([[sample[0] for sample in negative_couples]])
text_X2.append([[sample[1] for sample in negative_couples]])
text_y.extend([0] * len(negative_couples))
'''
for sample in positive_couples:
    text_X1.append(sample[0])
    text_X2.append(sample[1])
    text_y.append(1)
for sample in negative_couples:
    text_X1.append(sample[0])
    text_X2.append(sample[1])
    text_y.append(0)

text_y = np.array(text_y)
text_X1 = np.array(text_X1)
text_X2 = np.array(text_X2)

#for reshape dividors of 30 had to be chosen (don't know if fitting for the task)
text_X1 = text_X1.reshape((len(negative_couples) + len(positive_couples), 31, 1, 1))
text_X2 = text_X2.reshape((len(negative_couples) + len(positive_couples), 31, 1, 1))

text_X1 = 1 - text_X1 / 255
text_X2 = 1 - text_X2 / 255

print("Text data: ", text_X1.shape, text_X2.shape, text_y.shape)

# train model
X1 = text_X1
X2 = text_X2
y  = text_y
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

#reshape array
r,c= test_data[0].shape

test_data = np.array(test_data)
test_data = test_data.reshape(len(test_data),r,c,3)
test_data = 1 - test_data/255

# Predict
pred = model.predict(test_data)

num = int(pred.shape[0]/3)
subjects = [key for key in data_by_part.keys()]

y = []

for key in subjects:
    y += [key for i in range(num)]
feat1 = pred[:,0]
feat2 = pred[:,1]
feat3 = pred[:,2]

# plot 3d scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(feat1, feat2, feat3, c=y, marker='.')
plt.show()