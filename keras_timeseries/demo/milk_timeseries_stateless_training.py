import pandas as pd
import os

from keras_timeseries.library.recurrent import StatelessLSTM

data_dir_path = '../data'
model_dir_path = '../models/monthly-milk-production'

dataframe = pd.read_csv(filepath_or_buffer=os.path.join(data_dir_path, 'monthly-milk-production-pounds-p.csv'), sep=',')

print(dataframe.head())

timeseries = dataframe.as_matrix(['MilkProduction']).T[0]

print(timeseries)

network = StatelessLSTM()

network.fit(timeseries, model_dir_path=model_dir_path, num_timesteps=6)




