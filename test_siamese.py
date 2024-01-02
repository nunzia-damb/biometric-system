import numpy as np
import sklearn.preprocessing
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.layers import Lambda
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, RocCurveDisplay, DetCurveDisplay
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Function to generate random sequences

# Function to generate pairs of sequences and labels
def generate_pairs(cut=-1):
    """
    Generate pairs of positive and negatives keystrokes split in train and test
    :return: X_test, X_train, y_test, y_train where
    """
    from parse_dataset import get_dataset
    X_train, X_test, y_train, y_test, shape = get_dataset(cut)
    return X_test, X_train, y_test, y_train, shape


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, _ = shapes
    print("shapes: ", shapes)
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0.0))
    return K.mean(y_true * square_pred + (1.0 - y_true) * margin_square)


def create_siamese_model(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    base_network = create_base_network(input_shape)

    # data_parser.user_data = embedding(data_parser.user_data[0].phrases[0])
    # mask = Masking(mask_value=0, input_shape=input_shape)
    masked_a = Masking(mask_value=0, input_shape=input_shape)(input_a)
    masked_b = Masking(mask_value=0, input_shape=input_shape)(input_b)

    feature_vector_A = base_network(masked_a)
    feature_vector_B = base_network(masked_b)

    # concat = Concatenate()([feature_vector_A, feature_vector_B])
    # dense = Dense(64, activation='relu')(concat)

    distance = Lambda(euclidean_distance, output_shape=(8))([feature_vector_A, feature_vector_B])

    # sig = Dense(1, activation='sigmoid')(distance)

    model = Model(inputs=[input_a, input_b], outputs=distance)

    return model


def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Dropout(0.4)(input)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(input, x)


# Reshape the data

X_test, X_train, y_test, y_train, shape = generate_pairs(100)

# Create siamese model
print(shape)
input_shape = shape
siamese_model = create_siamese_model(input_shape)

siamese_model.summary()

# Compile the model
siamese_model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

scaler = StandardScaler()

X_trainnn = np.array(X_train)
nsamples, nx, ny, nz = X_trainnn.shape
X_trainnn = X_trainnn.reshape((nsamples, nx * ny * nz))
X_trainnn = scaler.fit_transform(X_trainnn)
X_trainnn = X_trainnn.reshape((nsamples, nx, ny, nz))

X_test = np.array(X_test)
nsamples, nx, ny, nz = X_test.shape
X_test = X_test.reshape((nsamples, nx * ny * nz))
X_test = scaler.fit_transform(X_test)
X_test = X_test.reshape((nsamples, nx, ny, nz))

a, b = X_trainnn[:, 0, :, :], X_trainnn[:, 1, :, :]

a_test, b_test = X_test[:, 0, :, :], X_test[:, 1, :, :]

# Train the model
siamese_model.fit([a, b], y_train, epochs=1, batch_size=8, validation_data=([a_test, b_test], y_test),
                  validation_freq=1, use_multiprocessing=True, workers=7, verbose=1)
# Evaluate the model on the test set

evaluation = siamese_model.evaluate([a_test, b_test], y_test, verbose=0)

print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

prediction = siamese_model.predict([a_test, b_test], verbose=0).ravel()
RocCurveDisplay.from_predictions(y_test, prediction)
DetCurveDisplay.from_predictions(y_test, prediction)
plt.show()

cm = confusion_matrix(y_test, np.argmax(siamese_model.predict([a_test, b_test], verbose=0), axis=-1))
print(cm)
ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

## Display the visualization of the Confusion Matrix.
plt.show()
prediction = siamese_model.predict([a_test, b_test], verbose=0).ravel()

fpr, tpr, thresholds = roc_curve(y_test, prediction)
auc_keras = auc(fpr, tpr)
print(auc_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Siamese (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

pass
