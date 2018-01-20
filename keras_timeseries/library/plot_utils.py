from matplotlib import pyplot as plt
import numpy as np
import itertools


def create_timeseries_plot(original, predicted, model_name):
    plt.title('Accuracy and Loss (' + model_name + ')')
    
    plt.plot(original, color='g', label='Actual')
    plt.plot(predicted, color='b', label='Predicted')
    plt.legend(loc='best')

    plt.tight_layout()


def plot_timeseries(original, predicted, model_name):
    create_timeseries_plot(original, predicted, model_name)
    plt.show()


def plot_and_save_timeseries(original, predicted, model_name, file_path):
    create_timeseries_plot(original, predicted, model_name)
    plt.savefig(file_path)
