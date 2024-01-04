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
    checkpoint_path_best_only = "best_chkpt" + os.sep + "cp-best.ckpt"

    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  save_best_only=False,
                                  monitor='val_accuracy',
                                  mode='max',
                                  verbose=1)
    cp_callback_best_only = ModelCheckpoint(filepath=checkpoint_path_best_only,
                                            save_weights_only=True,
                                            save_best_only=True,
                                            monitor='val_accuracy',
                                            mode='max',
                                            verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    return [history, cp_callback, cp_callback_best_only]


def normalize_dataset(X_train, X_test):
    scaler = StandardScaler()
    x_trainnn = np.array(X_train)
    x_testtt = np.array(X_test)

    nsamples_test, nx, ny, nz = x_testtt.shape
    nsamples_train, nx, ny, nz = x_trainnn.shape

    x_trainnn = x_trainnn.reshape((nsamples_train, nx * ny * nz))
    x_testtt = x_testtt.reshape((nsamples_test, nx * ny * nz))

    scaler = scaler.fit(np.concatenate((x_trainnn, x_testtt)))

    # apply normalizations
    x_trainnn = scaler.transform(x_trainnn)
    x_trainnn = x_trainnn.reshape((nsamples_train, nx, ny, nz))

    x_testtt = scaler.transform(x_testtt)
    x_testtt = x_testtt.reshape((nsamples_test, nx, ny, nz))

    # separate siamese keystrokes
    _a, _b = x_trainnn[:, 0, :, :], x_trainnn[:, 1, :, :]

    _a_test, _b_test = x_testtt[:, 0, :, :], x_testtt[:, 1, :, :]

    return _a, _b, _a_test, _b_test


NUM_PAIRS = 100
X_test, X_train, y_test, y_train, shape = generate_pairs(NUM_PAIRS)

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
    batch_size = 100
    steps_per_epoch = min(a.shape[0] // batch_size, 200)
    # Train the model
    siamese_model.fit([a, b], y_train, epochs=10000, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                      validation_data=([a_test, b_test], y_test), callbacks=load_callbacks(),
                      validation_freq=1, use_multiprocessing=True, workers=7, verbose=1, shuffle=True)
    # Evaluate the model on the test set

    evaluation = siamese_model.evaluate([a_test, b_test], y_test, verbose=0)

    print("Test Loss:", evaluation[0])
    print("Test Accuracy:", evaluation[1])

    from plot_checkpoint import report

    report(siamese_model, a_test=a_test, b_test=b_test, y_test=y_test)
pass
