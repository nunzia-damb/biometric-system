import tensorflow as tf
from keras.src.layers import Flatten
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal
from sklearn.model_selection import train_test_split
import numpy as np
from plot_checkpoint import *
from parse_dataset import get_dataset
from test_siamese import generate_pairs

tf.random.set_seed(42069)


# Create Siamese model
def create_siamese_model(input_shape):
    def create_feature_extractor(i):
        input_layer = Input(shape=i)
        x_ = Dense(128, activation='relu')(input_layer)
        x_ = Dense(64, activation='relu')(x_)
        x_ = Dense(32, activation='relu')(x_)
        o = Dense(1, activation='sigmoid')(x_)
        return Model(inputs=input_layer, outputs=o)

    def create_decision_module(i):
        x_ = Dense(64, activation='relu')(i)
        x_ = Dense(32, activation='relu')(x_)
        o = Dense(1, activation='sigmoid')(x_)
        return o

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Create a shared feature extractor
    feature_extractor = create_feature_extractor(input_shape)

    # Connect both inputs to the shared feature extractor
    feature_vector_a = feature_extractor(input_a)
    feature_vector_b = feature_extractor(input_b)

    # Calculate L1 distance
    l1_distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([feature_vector_a, feature_vector_b])

    # Create decision module
    decision_module = create_decision_module(l1_distance)

    # Concatenate feature vectors and decision module output
    merged = Concatenate()([feature_vector_a, feature_vector_b, decision_module])
    # x = Dense(64, activation='relu')(merged)
    # x = Dense(32, activation='relu')(x)
    # x = Flatten()(merged)
    output_layer = Dense(1, activation='sigmoid')(merged)
    # Final model
    model = Model(inputs=[input_a, input_b], outputs=output_layer)

    return model


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
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    return [history, cp_callback_best_only]


def normalize_dataset(X_train, X_test, scaler, fit=True):
    x_trainnn = np.array(X_train)
    x_testtt = np.array(X_test)

    nsamples_test, nx, ny, nz = x_testtt.shape
    nsamples_train, nx, ny, nz = x_trainnn.shape

    x_trainnn = x_trainnn.reshape((nsamples_train, nx * ny * nz))
    x_testtt = x_testtt.reshape((nsamples_test, nx * ny * nz))
    if fit:
        scaler = scaler.fit(np.concatenate((x_trainnn, x_testtt)))

    # apply normalizations
    x_trainnn = scaler.transform(x_trainnn)
    x_trainnn = x_trainnn.reshape((nsamples_train, nx, ny, nz))

    x_testtt = scaler.transform(x_testtt)
    x_testtt = x_testtt.reshape((nsamples_test, nx, ny, nz))

    # separate siamese keystrokes
    _a, _b = x_trainnn[:, 0, :, :], x_trainnn[:, 1, :, :]

    _a_test, _b_test = x_testtt[:, 0, :, :], x_testtt[:, 1, :, :]

    return _a, _b, _a_test, _b_test, scaler


NUM_PAIRS = 1000
X_test, X_train, y_test, y_train, shape = generate_pairs(NUM_PAIRS)

standard_scaler = StandardScaler()
a, b, a_test, b_test, standard_scaler = normalize_dataset(X_train, X_test, standard_scaler)

pass

if __name__ == '__main__':
    del X_test, X_train

    # Create siamese model
    print('input shape', shape)
    input_shape = shape

    # Create Siamese model for web
    siamese_model = create_siamese_model(input_shape)
    siamese_model.summary()

    # Compile the model
    siamese_model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['val_accuracy'])

    callbacks = load_callbacks()

    print(a.shape, b.shape, y_train.shape)
    # Train the model
    siamese_model.fit([a, b], y_train, epochs=20, batch_size=100,
                      validation_data=([a_test, b_test], y_test),
                      callbacks=callbacks)

    # Save or use the model as needed
    del a, b, a_test, b_test, y_test, y_train  # delete heaviest data
    print('evaluation on never seen dataset')
    X_train, X_test, y_train, y_test, _ = get_dataset(cut=-1, validation=True)
    X_test = np.concatenate((X_train, X_test))
    y_test = np.concatenate((y_train, y_test))
    X_test = X_test[:, :, :70, :]
    _, _, a_test, b_test, standard_scaler = normalize_dataset(X_test, X_test, standard_scaler, fit=False)

    evaluation = siamese_model.evaluate([a_test, b_test], y_test, verbose=0)
    p = siamese_model.predict([a_test, b_test])
    print(p)
    print("Test Loss:", evaluation[0])
    print("Test Accuracy:", evaluation[1])

    report(siamese_model, a_test=a_test, b_test=b_test, y_test=y_test, history=callbacks[0])
pass
