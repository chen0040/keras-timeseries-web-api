import quandl

from keras_timeseries.library.plot_utils import plot_timeseries
from keras_timeseries.library.recurrent import StatelessLSTM

mydata = quandl.get("WIKI/MSFT")
# mydata = quandl.get("WIKI/MSFT", returns="numpy")

print(mydata.head())

timeseries = mydata.as_matrix(['Close']).T[0]

print(timeseries.shape)

network = StatelessLSTM()

model_dir_path = './models/stocker'

network.load_model(model_dir_path=model_dir_path)

predicted_list = []
actual_list = []
timesteps = 6
for i in range(timeseries.shape[0] - timesteps - 1):
    X = timeseries[i:i + timesteps].T
    predicted = network.predict(X)
    actual = timeseries[i + timesteps + 1]
    predicted_list.append(predicted)
    actual_list.append(actual)
    print('predicted: ' + str(predicted) + ' actual: ' + str(actual))

plot_timeseries(actual_list, predicted_list, StatelessLSTM.model_name)