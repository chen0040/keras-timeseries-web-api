import pandas as pd
import os
import numpy as np

from keras_timeseries.library.recurrent import StatefulLSTM

data_dir_path = '../data'
model_dir_path = '../models/monthly-milk-production'

dataframe = pd.read_csv(filepath_or_buffer=os.path.join(data_dir_path, 'monthly-milk-production-pounds-p.csv'), sep=',')

print(dataframe.head())

timeseries = dataframe.as_matrix(['MilkProduction']).T[0]

print(timeseries)

network = StatefulLSTM()

network.load_model(model_dir_path=model_dir_path)

timesteps = 6
for i in range(timeseries.shape[0] - timesteps - 1):
    X = timeseries[i:i + timesteps].T
    predicted = network.predict(X)
    actual = timeseries[i + timesteps + 1]
    print('predicted: ' + str(predicted) + ' actual: ' + str(actual))




