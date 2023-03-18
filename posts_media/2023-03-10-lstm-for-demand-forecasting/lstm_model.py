import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (10, 3)
plt.rcParams['figure.dpi'] = 150


class EarlyStoping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.001:
            print('Early stopping 0.001 reached. Training stopped!')
            self.model.stop_training = True

def create_uncompiled_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=1024, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
        tf.keras.layers.Dense(units=1)
    ])
    
    return model

def creat_model():
    tf.random.set_seed(51)
    model = create_uncompiled_model()
    model.compile(loss=tf.keras.losses.Huber(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])
    
    return model

def compute_metrics(true_series, predicted_series):
    mse = tf.keras.metrics.mean_squared_error(true_series, predicted_series).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, predicted_series).numpy()
    
    return mse, mae

def model_forecast(model, series, window_size):
    """Forecast the future values"""
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(32).prefetch(1)
    forecast = model.predict(dataset)
    
    return forecast

def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()