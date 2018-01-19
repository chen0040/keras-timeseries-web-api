import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import math
import pickle

DATA_DIR = './data'
MODEL_DIR = './models'
FILE_NAME = 'monthly-milk-production-pounds-p.csv'

dataframe = pd.read_csv(filepath_or_buffer=os.path.join(DATA_DIR, FILE_NAME), sep=',')

timeseries = dataframe.as_matrix(['MilkProduction']).T[0]

scaler = MinMaxScaler()
timeseries = scaler.fit_transform(timeseries)
pickle.dump(scaler, open(os.path.join(MODEL_DIR, 'monthly-milk-production-stateless-scaler.pickle'), 'wb'))

NUM_TIMESTEPS = 6 # use last 6 months data to predict the next value
BATCH_SIZE = 12
HIDDEN_UNITS = 64
NUM_EPOCHES = 100

context = dict()
context['timesteps'] = NUM_TIMESTEPS
context['scaler_min'] = scaler.data_min_
context['scaler_max'] = scaler.data_max_
context['scaler_range'] = scaler.data_range_
np.save(os.path.join(MODEL_DIR, 'monthly-milk-production-stateless-context.npy'), context)

model = Sequential()
model.add(LSTM(units=HIDDEN_UNITS, input_shape=(NUM_TIMESTEPS, 1), return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

X = np.zeros(shape=(timeseries.shape[0] - NUM_TIMESTEPS-1, NUM_TIMESTEPS))
Y = np.zeros(shape=(timeseries.shape[0] - NUM_TIMESTEPS-1, 1))
for i in range(timeseries.shape[0] - NUM_TIMESTEPS - 1):
    X[i] = timeseries[i:i+NUM_TIMESTEPS].T
    Y[i] = timeseries[i+NUM_TIMESTEPS+1]

X = np.expand_dims(X, axis=2)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHES, verbose=1, validation_data=(Xtest, Ytest))

score, _ = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE, verbose=1)

print('root mean squared error: ', math.sqrt(score))

with open('models/monthly-milk-production-stateless-architecture.json', 'w') as file:
    file.write(model.to_json())
model.save_weights('models/monthly-milk-production-stateless-weights.h5')