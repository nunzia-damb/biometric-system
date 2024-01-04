

def load_model(epoch):
    # Load the previously saved weights
    mymodel = create_siamese_model(input_shape)
    mymodel.load_weights(f'checkpoints/cp-{epoch:04d}.ckpt')
    return mymodel


def report(model, *, a_test, b_test, y_test, threshold=0.5):
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.metrics import RocCurveDisplay, DetCurveDisplay, confusion_matrix, classification_report
    import seaborn as sns

    prediction = (model.predict([a_test, b_test], verbose=0).ravel()).astype(np.float64)
    RocCurveDisplay.from_predictions(y_test, prediction)
    DetCurveDisplay.from_predictions(y_test, prediction)
    plt.show()

    prediction = (prediction > threshold).astype(np.float64)
    cm = confusion_matrix(y_test, prediction)
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
    cr = classification_report(y_test, prediction, labels=None, target_names=['0', '1'], digits=4)
    print(cr)


if __name__ == '__main__':
    from test_siamese import y_test, a_test, b_test, shape as input_shape, create_siamese_model

    chkpt = int(input('insert checkpoint '))
    siamese = create_siamese_model(input_shape)
    report(siamese, b_test=b_test, a_test=a_test, y_test=y_test)
