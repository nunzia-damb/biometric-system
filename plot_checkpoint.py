from sklearn.metrics import roc_curve, det_curve


def load_model(epoch):
    # Load the previously saved weights
    mymodel = create_siamese_model(input_shape)
    if epoch >= 0:
        mymodel.load_weights(f'checkpoints/cp-{epoch:04d}.ckpt')
    else:
        mymodel.load_weights(f'best_chkpt/cp-best.ckpt')

    return mymodel


import matplotlib.pyplot as plt
import numpy as np


def plot_err(frr, far, thresholds):
    """
    Plot FRR and FAR curves and identify Equal Error Rate (ERR).

    Parameters:
    - frr (numpy array): False Rejection Rate values.
    - far (numpy array): False Acceptance Rate values.
    - thresholds (numpy array): Decision thresholds.

    Returns:
    - err (float): Equal Error Rate (ERR).
    """

    # Plot FRR and FAR curves
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, frr, label='FRR')
    plt.plot(thresholds, far, label='FAR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('FRR and FAR Curves')
    plt.legend()
    plt.grid(True)

    # Find the index where FRR and FAR are closest
    min_err_index = np.argmin(np.abs(frr - far))
    err = (frr[min_err_index] + far[min_err_index]) / 2

    # Plot ERR point
    plt.scatter(thresholds[min_err_index], err, color='red', label=f'ERR: {err:.4f}')
    plt.legend()

    plt.show()

    return err


def report(model, *, a_test, b_test, y_test, threshold=0.5, history=None, save=False):
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.metrics import RocCurveDisplay, DetCurveDisplay, confusion_matrix, classification_report
    import seaborn as sns

    prediction = (model.predict([a_test, b_test], verbose=0).ravel()).astype(np.float64)

    RocCurveDisplay.from_predictions(y_test, prediction)
    DetCurveDisplay.from_predictions(y_test, prediction)
    if not save:
        plt.show()
    else:
        plt.savefig(f'')

    prediction_bool = (prediction > threshold).astype(np.bool8)
    cm = confusion_matrix(y_test, prediction_bool)
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
    cr = classification_report(y_test, prediction_bool, labels=None, target_names=['Positives', 'Negatives'], digits=4)
    print(cr)

    plt.show()

    far, frr, thresholds = det_curve(y_test, prediction)
    err = plot_err(frr=frr, far=far, thresholds=thresholds)
    print(f'Equal Error Rate (ERR): {err:.4f}')

    if history is not None:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    from test_siamese import create_siamese_model, standard_scaler
    from parse_dataset import get_dataset

    chkpt = int(input('insert checkpoint number; type -1 for the best of all '))

    # generate a never seen dataset
    x_val, y_true, input_shape = get_dataset(cut=-1, validation=True)
    a_test, b_test = x_val[:, 0, :70, :], x_val[:, 1, :70, :]
    input_shape = a_test.shape[1:]

    # load model
    siamese = load_model(chkpt)
    report(siamese, b_test=b_test, a_test=a_test, y_test=y_true)
