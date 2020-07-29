from matplotlib import pyplot as plt
from pathlib import Path

def save_history_plot(history, model_preamble, folder_name_logging):

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('{} model accuracy'.format(model_preamble))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(Path(folder_name_logging, "{}_model_training_accuracy.png".format(model_preamble))))
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('{} model loss'.format(model_preamble))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(Path(folder_name_logging, "{}_model_training_loss.png".format(model_preamble))))
    plt.clf()