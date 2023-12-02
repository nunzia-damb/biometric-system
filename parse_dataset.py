import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil
from random import random, randint, seed
import random
import pickle, itertools, sklearn, pandas as pd, seaborn as sn
from scipy.spatial import distance
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.utils.vis_utils import plot_model
from scipy import spatial
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

#import model
from main import train_keystrokes_rec

#parse excel
data_frame = pd.read_csv('data/DSL-StrongPasswordData.csv')

data_by_part = {}

#dictionary with partecipants as keys and list of metrics as values
#can be used as positive samples
key = ''
columns = list(data_frame.columns)[3:]
for i in data_frame.index:
    row = []
    for tag in columns[1:]:
        row.append(data_frame[tag][i])
    if data_frame['subject'][i] == key:
        data_by_part[key].append(row)
    else:
        key = data_frame['subject'][i]
        data_by_part[key] = [row]

#Test written
#takes as data for training first 20 entries pf every subject
positive_couples = []
for key in data_by_part:
    positive_couples += list(itertools.combinations(data_by_part[key][:20],2))
    

negative_couples = []
for key1 in data_by_part:
    for key2 in data_by_part:
        if key2!=key1:
            negative_couples+=list(itertools.product(data_by_part[key1][:20],data_by_part[key2][:20]))
            
text_X1 = []
text_X2 = []
text_y = []


# Create pairs of images and set target label for them. Target output is 1 if pair of images have same color else it is 0
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
text_X1 = text_X1.reshape((len(negative_couples) + len(positive_couples), 3, 2, 5))
text_X2 = text_X2.reshape((len(negative_couples) + len(positive_couples), 3, 2, 5))

text_X1 = 1 - text_X1/255
text_X2 = 1 - text_X2/255

print("Text data: ", text_X1.shape, text_X2.shape, text_y.shape)

#Save test data
f = open(os.getcwd()+"/test_texts.pk1",'wb')
pickle.dump([data_by_part[key][:20] for key in data_by_part],f)
f.close()

#train model
train_keystrokes_rec(text_X1,text_X2,text_y)