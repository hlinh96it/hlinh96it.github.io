import pandas as pd
import numpy as np
import tensorflow as tf

from generate_data import *
from lstm_model import *

import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (10, 3)
plt.rcParams['figure.dpi'] = 150

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Let's save the parameters of our time series in the dataclass
class PARAMETERS:
    SPLIT_TIME = 1100 # on day 1100 the training period will end. The rest will belong to the validation set
    WINDOW_SIZE = 20 # how many data points will we take into account to make our prediction
    BATCH_SIZE = 32 # how many items will we supply per batch
    SHUFFLE_BUFFER_SIZE = 1000 # we need this parameter to define the Tensorflow sample buffer


time, series = generate_time_series()
dataset = creat_window_dataset(series)
train_x, train_y, val_x, val_y = train_val_split(time, series)

early_stopping = EarlyStoping()
model = creat_model()
history = model.fit(dataset, epochs=1, callbacks=[early_stopping])

# plot MAE and loss
plt.figure(figsize=(10, 3))
plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.show()

# Prediction on the whole series
all_forecast = model_forecast(model, series, PARAMETERS.WINDOW_SIZE).squeeze()

# Validation portion
val_forecast = all_forecast[PARAMETERS.SPLIT_TIME - PARAMETERS.WINDOW_SIZE:-1]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(val_y, label="validation set")
plt.plot(val_forecast, label="predicted")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()
plt.show()


