from keras.models import Model
from keras.models import model_from_json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


class MilkStateless(object):
    model = None
    scaler = None

    def __init__(self):
        self.model = model_from_json(
            open('../keras_timeseries/models/monthly-milk-production-stateless-architecture.json', 'r').read())
        self.model.load_weights('../keras_timeseries/models/monthly-milk-production-stateless-weights.h5')
        self.context = np.load('../keras_timeseries/models/monthly-milk-production-stateless-context.npy').item()
        self.scaler = pickle.load(
            open('../keras_timeseries/models/monthly-milk-production-stateless-scaler.pickle', 'rb'))

    def evaluate(self, input):
        timeseries = self.scaler.transform(input)
        timesteps = self.context['timesteps']
        X = np.zeros(shape=(timeseries.shape[0] - timesteps - 1, timesteps))
        Y = np.zeros(shape=(timeseries.shape[0] - timesteps - 1, 1))
        for i in range(timeseries.shape[0] - timesteps - 1):
            X[i] = timeseries[i:i + timesteps].T
            Y[i] = timeseries[i + timesteps + 1]
        X = np.expand_dims(X, axis=2)
        Ypredict = self.model.predict(X)
        return Ypredict, Y

    def test_run(self):
        DATA_DIR = '../keras_timeseries/data'
        FILE_NAME = 'monthly-milk-production-pounds-p.csv'

        dataframe = pd.read_csv(filepath_or_buffer=os.path.join(DATA_DIR, FILE_NAME), sep=',')

        timeseries = dataframe.as_matrix(['MilkProduction']).T[0]

        Ypredict, Y = self.evaluate(timeseries)

        print(np.column_stack((Ypredict, Y)))
        return np.column_stack((Ypredict, Y))


if __name__ == '__main__':
    predictor = MilkStateless()
    predictor.test_run()
