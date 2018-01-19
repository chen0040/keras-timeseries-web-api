# keras-timeseries-web-api

Web api built on flask for keras-based timeseries forecasting using LSTM

# Implementation and Demo 

The implementation the stateful and stateless recurrent network can be found in 
[keras_timeseries/library/recurrent.py](keras_timeseries/library/recurrent.py)

The demo codes on how to use these recurrent networks can be found in the folder
[keras_timeseries/demo](keras_timeseries/demo)

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

Goto keras_timeseries/web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained classifiers:

* Stateful LSTM
* Stateless LSTM

