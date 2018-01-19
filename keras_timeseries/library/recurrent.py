from keras.layers import LSTM, Dense
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import math


class StatelessLSTM(object):
    model_name = 'stateless-lstm'

    def __init__(self):
        self.model = None
        self.scaler = None
        self.config = None

    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path + '/' + StatelessLSTM.model_name + '-config.npy'

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + '/' + StatelessLSTM.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_path(model_dir_path):
        return model_dir_path + '/' + StatelessLSTM.model_name + '-architecture.json'

    @staticmethod
    def get_scaler_path(model_dir_path):
        return model_dir_path + '/' + StatelessLSTM.model_name + '-scaler.pickle'

    def load_model(self, model_dir_path):
        config_file_path = StatelessLSTM.get_config_path(model_dir_path)
        weight_file_path = StatelessLSTM.get_weight_path(model_dir_path)
        architecture_file_path = StatelessLSTM.get_architecture_path(model_dir_path)
        scaler_file_path = StatelessLSTM.get_scaler_path(model_dir_path)

        self.model = model_from_json(
            open(config_file_path, 'r').read())
        self.model.load_weights(weight_file_path)
        self.config = np.load(architecture_file_path).item()
        self.scaler = pickle.load(
            open(scaler_file_path, 'rb'))

    def evaluate(self, timeseries):
        timeseries = self.scaler.transform(timeseries)
        timesteps = self.config['timesteps']
        X = np.zeros(shape=(timeseries.shape[0] - timesteps - 1, timesteps))
        Y = np.zeros(shape=(timeseries.shape[0] - timesteps - 1, 1))
        for i in range(timeseries.shape[0] - timesteps - 1):
            X[i] = timeseries[i:i + timesteps].T
            Y[i] = timeseries[i + timesteps + 1]
        X = np.expand_dims(X, axis=2)
        Ypredict = self.model.predict(X)
        return Ypredict, Y

    def create_model(self, num_timesteps, hidden_units=None):
        if hidden_units is None:
            hidden_units = 64

        model = Sequential()
        model.add(LSTM(units=hidden_units, input_shape=(num_timesteps, 1), return_sequences=False))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        print(model.summary())
        return model

    def fit(self, timeseries, model_dir_path, batch_size=None, epochs=None, num_timesteps=None):
        weight_file_path = StatelessLSTM.get_weight_path(model_dir_path)
        architecture_file_path = StatelessLSTM.get_architecture_path(model_dir_path)
        scaler_file_path = StatelessLSTM.get_scaler_path(model_dir_path)
        config_file_path = StatelessLSTM.get_config_path(model_dir_path)

        self.scaler = MinMaxScaler()
        timeseries = self.scaler.fit_transform(timeseries)
        pickle.dump(self.scaler, open(scaler_file_path, 'wb'))

        if num_timesteps is None:
            num_timesteps = 6  # use last 6 timestep data to predict the next value
        if batch_size is None:
            batch_size = 12
        if epochs is None:
            epochs = 100

        self.config = dict()
        self.config['timesteps'] = num_timesteps
        self.config['scaler_min'] = self.scaler.data_min_
        self.config['scaler_max'] = self.scaler.data_max_
        self.config['scaler_range'] = self.scaler.data_range_
        np.save(config_file_path, self.config)

        self.model = self.create_model(num_timesteps)
        with open(architecture_file_path, 'w') as file:
            file.write(self.model.to_json())

        X = np.zeros(shape=(timeseries.shape[0] - num_timesteps - 1, num_timesteps))
        Y = np.zeros(shape=(timeseries.shape[0] - num_timesteps - 1, 1))
        for i in range(timeseries.shape[0] - num_timesteps - 1):
            X[i] = timeseries[i:i + num_timesteps].T
            Y[i] = timeseries[i + num_timesteps + 1]

        X = np.expand_dims(X, axis=2)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

        checkpoint = ModelCheckpoint(weight_file_path)
        self.model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                       callbacks=[checkpoint],
                       validation_data=(Xtest, Ytest))

        score, _ = self.model.evaluate(Xtest, Ytest, batch_size=batch_size, verbose=1)

        print('root mean squared error: ', math.sqrt(score))

        self.model.save_weights(weight_file_path)

    def test_run(self, timeseries):
        Ypredict, Y = self.evaluate(timeseries)

        return np.column_stack((Ypredict, Y))
    
    
class StatefulLSTM(object):
    model_name = 'stateful-lstm'

    def __init__(self):
        self.model = None
        self.scaler = None
        self.config = None

    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path + '/' + StatefulLSTM.model_name + '-config.npy'

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + '/' + StatefulLSTM.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_path(model_dir_path):
        return model_dir_path + '/' + StatefulLSTM.model_name + '-architecture.json'

    @staticmethod
    def get_scaler_path(model_dir_path):
        return model_dir_path + '/' + StatefulLSTM.model_name + '-scaler.pickle'

    def load_model(self, model_dir_path):
        config_file_path = StatefulLSTM.get_config_path(model_dir_path)
        weight_file_path = StatefulLSTM.get_weight_path(model_dir_path)
        architecture_file_path = StatefulLSTM.get_architecture_path(model_dir_path)
        scaler_file_path = StatefulLSTM.get_scaler_path(model_dir_path)

        self.model = model_from_json(
            open(config_file_path, 'r').read())
        self.model.load_weights(weight_file_path)
        self.config = np.load(architecture_file_path).item()
        self.scaler = pickle.load(
            open(scaler_file_path, 'rb'))

    def evaluate(self, timeseries):
        timeseries = self.scaler.transform(timeseries)
        timesteps = self.config['timesteps']
        X = np.zeros(shape=(timeseries.shape[0] - timesteps - 1, timesteps))
        Y = np.zeros(shape=(timeseries.shape[0] - timesteps - 1, 1))
        for i in range(timeseries.shape[0] - timesteps - 1):
            X[i] = timeseries[i:i + timesteps].T
            Y[i] = timeseries[i + timesteps + 1]
        X = np.expand_dims(X, axis=2)
        Ypredict = self.model.predict(X)
        return Ypredict, Y

    def create_model(self, num_timesteps, batch_size, hidden_units=None):
        if hidden_units is None:
            hidden_units = 64

        model = Sequential()
        model.add(LSTM(units=hidden_units, batch_input_shape=(BATCH_SIZE, num_timesteps, 1), return_sequences=False,
                       stateful=True))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        print(model.summary())
        return model

    def fit(self, timeseries, model_dir_path, batch_size, epochs=None, num_timesteps=None):
        weight_file_path = StatefulLSTM.get_weight_path(model_dir_path)
        architecture_file_path = StatefulLSTM.get_architecture_path(model_dir_path)
        scaler_file_path = StatefulLSTM.get_scaler_path(model_dir_path)
        config_file_path = StatefulLSTM.get_config_path(model_dir_path)

        self.scaler = MinMaxScaler()
        timeseries = self.scaler.fit_transform(timeseries)
        pickle.dump(self.scaler, open(scaler_file_path, 'wb'))

        if num_timesteps is None:
            num_timesteps = 6  # use last 6 timestep data to predict the next value
        if batch_size is None:
            batch_size = 12
        if epochs is None:
            epochs = 100

        self.config = dict()
        self.config['timesteps'] = num_timesteps
        self.config['scaler_min'] = self.scaler.data_min_
        self.config['scaler_max'] = self.scaler.data_max_
        self.config['scaler_range'] = self.scaler.data_range_
        self.config['batch_size'] = batch_size
        np.save(config_file_path, self.config)

        self.model = self.create_model(num_timesteps)
        with open(architecture_file_path, 'w') as file:
            file.write(self.model.to_json())

        X = np.zeros(shape=(timeseries.shape[0] - num_timesteps - 1, num_timesteps))
        Y = np.zeros(shape=(timeseries.shape[0] - num_timesteps - 1, 1))
        for i in range(timeseries.shape[0] - num_timesteps - 1):
            X[i] = timeseries[i:i + num_timesteps].T
            Y[i] = timeseries[i + num_timesteps + 1]

        X = np.expand_dims(X, axis=2)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

        # must be multiple of batch size
        trainSize = int(len(Xtrain) // batch_size)
        trainSize = trainSize * batch_size

        testSize = int(len(Xtest) // batch_size)
        testSize = testSize * batch_size

        checkpoint = ModelCheckpoint(weight_file_path)
        self.model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                       callbacks=[checkpoint],
                       validation_data=(Xtest, Ytest))

        score, _ = self.model.evaluate(Xtest, Ytest, batch_size=batch_size, verbose=1)

        print('root mean squared error: ', math.sqrt(score))

        self.model.save_weights(weight_file_path)

    def test_run(self, timeseries):
        Ypredict, Y = self.evaluate(timeseries)

        return np.column_stack((Ypredict, Y))


def main():
    data_dir_path = '../data'
    model_dir_path = '../models/monthly-milk-production'
    data_file_path = os.path.join(data_dir_path, 'monthly-milk-production-pounds-p.csv')
    dataframe = pd.read_csv(filepath_or_buffer=data_file_path, sep=',')

    predictor = StatelessLSTM()
    predictor.load_model(model_dir_path)

    timeseries = dataframe.as_matrix(['MilkProduction']).T[0][0:43]  # 36 is the multiple of the batch size
    predictor.test_run(timeseries)


if __name__ == '__main__':
    main()
