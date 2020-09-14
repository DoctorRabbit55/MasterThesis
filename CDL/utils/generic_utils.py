from matplotlib import pyplot as plt
from pathlib import Path

def save_history_plot(history, model_preamble, folder_name_logging, keys):

    history = history.history

    for key in keys:
        plt.plot(history[key])
        plt.plot(history['val_' + key])
        plt.title('{} model {}'.format(model_preamble, key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(str(Path(folder_name_logging, "{}_model_training_{}.png".format(model_preamble, key))))
        plt.clf()        