import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.layers import Lambda


# Function to generate random sequences

# Function to generate pairs of sequences and labels
def generate_pairs():
    """
    Generate pairs of positive and negatives keystrokes split in train and test
    :return: X_test, X_train, y_test, y_train where
    """
    from parse_dataset import X_test, X_train, y_test, y_train, shape
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
    x = Dense(128, activation='relu')(input)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    return Model(input, x)


# Reshape the data

X_test, X_train, y_test, y_train, shape = generate_pairs()

# Create siamese model
print(shape)
input_shape = shape
siamese_model = create_siamese_model(input_shape)

siamese_model.summary()

# Compile the model
siamese_model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


X_trainnn = np.array(X_train)
a = X_trainnn[:, 0, :, :]
b = X_trainnn[:, 1, :, :]

# Train the model
siamese_model.fit([a, b], y_train, epochs=10, batch_size=8, use_multiprocessing=True, workers=7, verbose=1)
# Evaluate the model on the test set

X_test = np.array(X_test)
a = X_test[:, 0, :, :]
b = X_test[:, 1, :, :]

evaluation = siamese_model.evaluate([a,b], y_test, verbose=0)

print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])


prediction = siamese_model.predict([a,b], verbose=0)

pass


