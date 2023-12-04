from mpl_toolkits.mplot3d import Axes3D
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, pickle
import sklearn, pandas as pd, seaborn as sn
from keras.models import Model, load_model, Sequential
from keras import backend as K
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')
from parse_dataset import data_by_part

#load models
model = load_model(os.getcwd()+"/text_encoder.h5")
siamese_model = load_model(os.getcwd()+"/text_siamese_model.h5")

#load test data
f = open(os.getcwd()+"/test_texts.pkl", 'rb')
test_texts = pickle.load(f)
f.close()

#reshape array
r,c= test_texts[0].shape

test_texts = np.array(test_texts)
test_texts = test_texts.reshape(len(test_texts),r,c,3)
test_texts = 1 - test_texts/255

# Predict
pred = model.predict(test_texts)

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