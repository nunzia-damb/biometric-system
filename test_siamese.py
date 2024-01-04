import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Lambda
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


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
    def create_feature_extractor(input_shape):
        input_layer = Input(shape=input_shape)
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        # x = Dense(16, activation='relu')(x)
        # output_layer = Dense(8, activation='relu')(x)
        return Model(inputs=input_layer, outputs=x)

    def create_decision_module(input_layer):
        x = Dense(64, activation='relu')(input_layer)
        x = Dense(32, activation='relu')(x)
        # x = Dense(16, activation='relu')(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        return output_layer

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # base_network = Sequential(
    #     Input(shape=input_shape),
    #     # x = Dropout(0.4)(input)
    #     # x = Dense(256, activation='relu')(input)
    #     # # x = Dropout(0.4)(x)
    #     # x = Dense(128, activation='relu')(x)
    #     # x = Dropout(0.4)(x)
    #     Dense(64, activation='relu'),
    #     # x = Dropout(0.4)(x)
    #     Dense(32, activation='relu'),
    #     # x = Dropout(0.4)(x)
    #     # x = Dense(16, activation='relu')(x)
    #     # # x = Dropout(0.4)(x)
    #     # x = Dense(8, activation='relu')(x)
    #     # x = Dropout(0.4)(x)
    #     Dense(1, activation='sigmoid'),
    #     Flatten(),
    #     name='shared_submodel'
    # )

    # data_parser.user_data = embedding(data_parser.user_data[0].phrases[0])
    # mask = Masking(mask_value=0, input_shape=input_shape)
    # masked_a = Masking(mask_value=0, input_shape=input_shape)(input_a)
    # masked_b = Masking(mask_value=0, input_shape=input_shape)(input_b)

    # Create a shared feature extractor
    feature_extractor = create_feature_extractor(input_shape)

    # Connect both inputs to the shared feature extractor
    feature_vector_a = feature_extractor(input_a)
    feature_vector_b = feature_extractor(input_b)

    # concat = Concatenate()([feature_vector_A, feature_vector_B])
    # dense = Dense(64, activation='relu')(concat)

    # distance = Lambda(euclidean_distance, output_shape=(8))([feature_vector_A, feature_vector_B])
    # concat = Concatenate()([feature_vector_A, feature_vector_B])
    l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([feature_vector_a, feature_vector_b])

    decision_module = create_decision_module(l1_distance)
    decision_module = Flatten()(decision_module)
    decision_module = Dense(1, activation='sigmoid')(decision_module)
    sm = Model(inputs=[input_a, input_b], outputs=decision_module)

    # dense = Dense(64, activation='relu')(l1_norm)
    # flattened = Flatten()(dense)
    # output = Dense(1, activation='sigmoid', name='classification_layer')(merged)

    # sig = Dense(1, activation='sigmoid')(distance)

    # model = Model(inputs=[input_a, input_b], outputs=output)

    return sm


def load_callbacks():
    import os
    from keras.src.callbacks import EarlyStopping, History, ModelCheckpoint

    history = History()
    checkpoint_path = "checkpoints" + os.sep + "cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  save_best_only=False,
                                  monitor='val_accuracy',
                                  mode='max',
                                  verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    return [history, cp_callback, early_stopping]

def normalize_dataset(X_train, X_test):
    scaler = StandardScaler()

    x_trainnn = np.array(X_train)
    nsamples, nx, ny, nz = x_trainnn.shape
    x_trainnn = x_trainnn.reshape((nsamples, nx * ny * nz))
    x_trainnn = scaler.fit_transform(x_trainnn)
    x_trainnn = x_trainnn.reshape((nsamples, nx, ny, nz))

    X_test = np.array(X_test)
    nsamples, nx, ny, nz = X_test.shape
    X_test = X_test.reshape((nsamples, nx * ny * nz))
    X_test = scaler.fit_transform(X_test)
    X_test = X_test.reshape((nsamples, nx, ny, nz))

    a, b = x_trainnn[:, 0, :, :], x_trainnn[:, 1, :, :]

    a_test, b_test = X_test[:, 0, :, :], X_test[:, 1, :, :]

    return a, b, a_test, b_test


X_test, X_train, y_test, y_train, shape = generate_pairs(100)

a, b, a_test, b_test = normalize_dataset(X_train, X_test)
del X_test, X_train

if __name__ == '__main__':
    # Create siamese model
    print(shape)
    input_shape = shape
    siamese_model = create_siamese_model(input_shape)

    siamese_model.summary()

    # Compile the model
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001, ), metrics=['accuracy'])


    # Train the model
    siamese_model.fit([a, b], y_train, epochs=10, batch_size=100, steps_per_epoch=200,
                      validation_data=([a_test, b_test], y_test), callbacks=load_callbacks(),
                      validation_freq=1, use_multiprocessing=True, workers=7, verbose=1, shuffle=True)
    # Evaluate the model on the test set

    evaluation = siamese_model.evaluate([a_test, b_test], y_test, verbose=0)

    print("Test Loss:", evaluation[0])
    print("Test Accuracy:", evaluation[1])

pass
