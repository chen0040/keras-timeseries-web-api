import quandl

from keras_timeseries.library.recurrent import StatelessLSTM

mydata = quandl.get("WIKI/MSFT")
# mydata = quandl.get("WIKI/MSFT", returns="numpy")

print(mydata.head())

timeseries = mydata.as_matrix(['Close']).T[0]

print(timeseries.shape)

network = StatelessLSTM()

output_dir_path = './models/stocker'
timesteps = 6
network.fit(timeseries, model_dir_path=output_dir_path, num_timesteps=timesteps, epochs=10)