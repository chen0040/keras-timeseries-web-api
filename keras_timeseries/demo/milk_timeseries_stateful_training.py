import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import math
import pickle

DATA_DIR = './data'
MODEL_DIR = './models'
FILE_NAME = 'monthly-milk-production-pounds-p.csv'
NUM_TIMESTEPS = 6 # use last 6 months data to predict the next value
BATCH_SIZE = 12
HIDDEN_UNITS = 64
NUM_EPOCHES = 100

dataframe = pd.read_csv(filepath_or_buffer=os.path.join(DATA_DIR, FILE_NAME), sep=',')

timeseries = dataframe.as_matrix(['MilkProduction']).T[0]

scaler = MinMaxScaler()
timeseries = scaler.fit_transform(timeseries)
pickle.dump(scaler, open(os.path.join(MODEL_DIR, 'monthly-milk-production-stateful-scaler.pickle'), 'wb'))

context = dict()
context['timesteps'] = NUM_TIMESTEPS
np.save(os.path.join(MODEL_DIR, 'monthly-milk-production-stateful-context.npy'), context)

model = Sequential()
model.add(LSTM(units=HIDDEN_UNITS, batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, 1), return_sequences=False, stateful=True))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

X = np.zeros(shape=(timeseries.shape[0] - NUM_TIMESTEPS-1, NUM_TIMESTEPS))
Y = np.zeros(shape=(timeseries.shape[0] - NUM_TIMESTEPS-1, 1))
for i in range(timeseries.shape[0] - NUM_TIMESTEPS - 1):
    X[i] = timeseries[i:i+NUM_TIMESTEPS].T
    Y[i] = timeseries[i+NUM_TIMESTEPS+1]

X = np.expand_dims(X, axis=2)

trainSize = 120 # 120 is multiple of BATCH_SIZE

Xtrain, Xtest, Ytrain, Ytest = X[0:trainSize], X[trainSize:], Y[0:trainSize], Y[trainSize:]
Ytest = Ytest[0:36] # 36 is multiple of BATCH_SIZE
Xtest = Xtest[0:36]

model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHES, verbose=1, validation_data=(Xtest, Ytest))

score, _ = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE, verbose=1)

print('root mean squared error: ', math.sqrt(score))

with open('models/monthly-milk-production-stateful-architecture.json', 'w') as file:
    file.write(model.to_json())
model.save_weights('models/monthly-milk-production-stateful-weights.h5')