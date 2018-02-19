# keras-timeseries-web-api

Web api built on flask for keras-based timeseries forecasting using LSTM

# Implementation

The implementation the stateful and stateless recurrent network can be found in 
[keras_timeseries/library/recurrent.py](keras_timeseries/library/recurrent.py)

The demo codes on how to use these recurrent networks can be found in the folder
[demo](demo)

# Usage

### Demo

The demo codes on the milk yield production is shown below:

```python
import pandas as pd
import os
from keras_timeseries.library.plot_utils import plot_timeseries
from keras_timeseries.library.recurrent import StatelessLSTM

data_dir_path = './data'
output_dir_path = './models/monthly-milk-production'

dataframe = pd.read_csv(filepath_or_buffer=os.path.join(data_dir_path, 'monthly-milk-production-pounds-p.csv'), sep=',')

print(dataframe.head())

timeseries = dataframe.as_matrix(['MilkProduction']).T[0]

print(timeseries)

network = StatelessLSTM()

# training 
timesteps = 6
network.fit(timeseries, model_dir_path=output_dir_path, num_timesteps=timesteps)

# predicting
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
```

As the milk yield production has a seasonal pattern of 12 months (each period has 12 data points), we
can also use stateful LSTM, which is shown below:

```python
import pandas as pd
import os
from keras_timeseries.library.plot_utils import plot_timeseries
from keras_timeseries.library.recurrent import StatefulLSTM

data_dir_path = './data'
model_dir_path = './models/monthly-milk-production'

dataframe = pd.read_csv(filepath_or_buffer=os.path.join(data_dir_path, 'monthly-milk-production-pounds-p.csv'), sep=',')

print(dataframe.head())

timeseries = dataframe.as_matrix(['MilkProduction']).T[0]

network = StatefulLSTM()

# training
batch_size = 12  # 12 is the period of the "cycle"
network.fit(timeseries, batch_size=batch_size, model_dir_path=model_dir_path, num_timesteps=6)

# predicting
predicted_list = []
actual_list = []
timesteps = 6
for i in range(timeseries.shape[0] - timesteps - 1):
    X = timeseries[:i + timesteps].T
    predicted = network.predict(X)
    actual = timeseries[i + timesteps + 1]
    predicted_list.append(predicted)
    actual_list.append(actual)
    print('predicted: ' + str(predicted) + ' actual: ' + str(actual))

plot_timeseries(actual_list, predicted_list, StatefulLSTM.model_name)
```


### Api Web Server

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

Goto demo_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained classifiers:

* Stateful LSTM
* Stateless LSTM

