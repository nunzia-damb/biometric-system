import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.src.layers import Lambda


# Function to generate random sequences

# Function to generate pairs of sequences and labels
def generate_pairs(num_pairs=1000, sequence_length=10):
    # pairs, labels = [], []
    # for _ in range(num_pairs):
    #     # Generate positive pair (sequences from the same class)
    #     sequence_a = generate_sequence(sequence_length)
    #     sequence_b = generate_sequence(sequence_length)
    #     pairs.append((sequence_a, sequence_b))
    #     labels.append(1.0)
    #
    #     # Generate negative pair (sequences from different classes)
    #     sequence_c = generate_sequence(sequence_length)
    #     pairs.append((sequence_a, sequence_c))
    #     labels.append(0.0)
    from parse_dataset import train_X1, train_X2, train_y, test_X1, test_X2, test_y
    # return np.array(pairs), np.array(labels)
    return train_X1, train_X2, train_y, test_X1, test_X2, test_y


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

    feature_vector_A = base_network(input_a)
    feature_vector_B = base_network(input_b)

    # concat = Concatenate()([feature_vector_A, feature_vector_B])
    # dense = Dense(64, activation='relu')(concat)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feature_vector_A, feature_vector_B])

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

train_X1, train_X2, train_y, test_X1, test_X2, test_y = generate_pairs()

# Create siamese model
print(train_X1.shape)
input_shape = train_X1.shape[1:]
siamese_model = create_siamese_model(input_shape)

siamese_model.summary()

# Compile the model
siamese_model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
siamese_model.fit([train_X1, train_X2], train_y, epochs=1, batch_size=32, use_multiprocessing=True, workers=7, verbose=1)
# Evaluate the model on the test set
evaluation = siamese_model.evaluate([test_X1, test_X2], test_y, verbose=0)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])


prediction = siamese_model.predict([test_X1, test_X2], verbose=0)

pass


