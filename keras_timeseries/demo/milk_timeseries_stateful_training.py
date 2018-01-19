import pandas as pd
import os

from keras_timeseries.library.recurrent import StatefulLSTM

data_dir_path = '../data'
model_dir_path = '../models/monthly-milk-production'

dataframe = pd.read_csv(filepath_or_buffer=os.path.join(data_dir_path, 'monthly-milk-production-pounds-p.csv'), sep=',')

print(dataframe.head())

timeseries = dataframe.as_matrix(['MilkProduction']).T[0]

network = StatefulLSTM()

batch_size = 12  # 12 is the period of the "cycle"
network.fit(timeseries, batch_size=batch_size, model_dir_path=model_dir_path, num_timesteps=6)
