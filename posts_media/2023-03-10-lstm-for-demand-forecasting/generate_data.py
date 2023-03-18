import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def trend(time, slope=0):
    return slope*time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.1, np.cos(season_time*6*np.pi), 2/np.exp(9*season_time))

def add_seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def add_noise(time, noise_level=1, seed=None):
    """Adds white noise to the series"""
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def generate_time_series(amplitude=50, noise_level=3):
    time = np.arange(4*365 + 1, dtype=np.float32)
    y_intercep, slope = 10, 0.005
    series = trend(time, slope) + y_intercep
    
    # add seasonality pattern
    series += add_seasonality(time, period=365, amplitude=amplitude)
    
    # add some noise
    series += add_noise(time, noise_level, seed=51)
    
    return time, series

def creat_window_dataset(series, window_size, batch_size, shuffle):
    """
    Creat time window as X training and predict Y
    Eg. If we want to predict yt based on previous 30 data, X.shape = 30
    """
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.map(lambda window: (window[: -1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset

def train_val_split(time, series, time_step):
    return time[: time_step], series[: time_step], time[time_step:], series[time_step:]