---
title: Understanding Long Short Term Memory (LSTM) from Theory to Python - Part 2
author: hoanglinh
categories: [Deep Learning]
tags: [time series forecasting, natural language processing]
math: true
img_path: posts_media/2023-03-05-lstm-from-theory-to-python/
---

In the [previous post](https://www.codewithlinh.com/posts/lstm-from-theory-to-python-part-1/), we introduced the concept of Long Short Term Memory (LSTM) and discussed its theoretical underpinnings. In this post, we will take a more practical approach and implement LSTMs in Python code. To better understand how LSTMs work under the hood, it can be helpful to implement them in Python code from scratch. This will allow you to see the inner workings of the LSTM and gain a deeper understanding of how it processes and stores information. Once you have a solid grasp of the fundamentals, you can then move on to implementing LSTMs in popular deep learning libraries such as TensorFlow and Keras.

> **Give me some ideas**: [Understanding Long Short Term Memory (LSTM) from Theory to Python - Part 1](https://www.codewithlinh.com/posts/lstm-from-theory-to-python-part-1/)
{: .prompt-info}

# Understand weight matrix in a cell

One important aspect of implementing LSTMs is understanding the shape of the weight matrices used in the LSTM model. However, identifying the matrix shape can be a difficult task. To help you better understand the process, in this post, we will create a visualization that will enable you to identify the shape of the weight matrices used in LSTMs with ease.

By understanding the shape of the weight matrices, you can gain a deeper understanding of how information is processed and stored within LSTMs. This will allow you to better understand how to fine-tune your LSTM models to suit your specific needs. So stay tuned as we walk you through the process of creating a visualization to help identify the shape of the weight matrices used in LSTMs.

![lstm-matrix.png](lstm-matrix.png)

Considering that we have a data, shape of `[batch_size, time_steps, number_features]` , $X_{t}$ is the input of time-step $t$ which is an array with the shape of `[batch_size, number_features]`, $h_{t-1}$ is the hidden state of previous time-step which is an array with the shape of`[batch_size, number_units]`, and $C_{t-1}$ is the cell state of previous time-step, which is an array with the shape of `[batch_size, num_units]`. In this case, we will concatenate inputs ($X_{t}$) and hidden state $h_{t-1}$ by column and multiple it with kernel (weight) matrix.

Each of the $W_{xi}, W_{xf}, W_{xc}$ and $W_{xo}$, is an array with the shape of `[number_features, number_units]` and, similarly, each of the $W_{hi}, W_{hf}, W_{hc}$ and $W_{ho}$ is an array with the shape of `[number_units, num_units]`. If we first concatenate each gate weight matrices, corresponding to input and hidden state, vertically, we will have separate $W_{i}, W_{c}, W_{f}$ and $W_{o}$ matrices, which each will have the shape of `[number_features + number_units, number_units]`. Then, if we concatenate $W_{i},  W_{c}, W_{f}, W_{o}$ matrices horizontally, we will have kernel (weights) matrix, which has shape `[number_features + number_units, 4 * number_units]`

> The number of LSTM units corresponds to the number of hidden neurons in the LSTM layer. They are responsible for processing and storing information across time steps in a sequence.
{: .prompt-info}

Here's an example of how to create the weight matrix for a LSTM cell in Python:

```
hidden_units -- number of hidden units
input_units -- number of input units
    
parameters['forget_gate_weights'] -- Weight matrix of the forget gate, shape (hidden_units, hidden_units + input_units)
parameters['forget_gate_bias']  -- Bias of the forget gate, shape (hidden_units, 1)
parameters['input_gate_weights'] -- Weight matrix of the update gate, shape (hidden_units, hidden_units + input_units)
parameters['input_gate_bias'] -- Bias of the update gate, shape (hidden_units, 1)
parameters['gate_weights'] -- Weight matrix of the first "tanh", shape (hidden_units, hidden_units + input_units)
parameters['gate_bias'] --  Bias of the first "tanh", shape (hidden_units, 1)
parameters['output_gate_weights'] -- Weight matrix of the output gate, shape (hidden_units, hidden_units + input_units)
parameters['output_gate_bias'] --  Bias of the output gate, shape (hidden_units, 1)
parameters['hidden_output_weights'] -- Weight matrix relating the hidden-state to the output, (n_y, hidden_units)
parameters['hidden_output_bias'] -- Bias relating the hidden-state to the output (n_y, 1)

```

```python
def create_weight_matrix(input_units, hidden_units):
    mean, std = 0, 0.01
    parameters = {}
    
    # lstm cell weights and save all weights into a dict
    parameters['forget_gate_weights'] = np.random.normal(loc=mean, scale=std,
                                           size=(input_units + hidden_units, hidden_units))
    parameters['forget_gate_bias'] = np.random.normal(loc=mean, scale=std, size=(hidden_units, 1))
    
    parameters['input_gate_weights'] = np.random.normal(loc=mean, scale=std,
                                           size=(input_units + hidden_units, hidden_units))
    parameters['input_gate_bias'] = np.random.normal(loc=mean, scale=std, size=(hidden_units, 1))
    
    parameters['output_gate_weights'] = np.random.normal(loc=mean, scale=std,
                                           size=(input_units + hidden_units, hidden_units))
    parameters['output_gate_bias'] = np.random.normal(loc=mean, scale=std, size=(hidden_units, 1))
    
    parameters['gate_weights'] = np.random.normal(loc=mean, scale=std,
                                           size=(input_units + hidden_units, hidden_units))
    parameters['gate_bias'] = np.random.normal(loc=mean, scale=std, size=(hidden_units, 1))
    
    parameters['hidden_output_weights'] = np.random.normal(loc=mean, scale=std, size=(hidden_units, hidden_units))
    parameters['hidden_output_bias'] = np.random.normal(loc=mean, scale=std, size=(hidden_units, 1))
    
    return parameters
```

In this implementation, we first initialize the weight matrix with random values using a Gaussian distribution with mean 0 and standard deviation of 0.01. We then split the weight matrix into 4 parts corresponding to the input gate, forget gate, cell state, and output gate.

To use this function, simply call it with the desired number of inputs and hidden units:

```python
num_inputs = 10
num_units = 128

parameters = create_weight_matrix(num_inputs, num_units)
```

Overall, implementing the LSTM Forward Cell in Python requires a solid understanding of the shape of the weight matrices used in the LSTM model. By understanding these shapes, you can better understand how information is processed and stored within the LSTM, allowing you to fine-tune your models to suit your specific needs.

> Note that this implementation assumes that the LSTM cell has a single layer. If you're working with a multi-layer LSTM, you'll need to modify the implementation to account for multiple layers.
{: .prompt-tip}

## Forward Cell in Python

The LSTM Forward Cell is an important component of the [Long Short Term Memory (LSTM)](https://www.codewithlinh.com/posts/lstm-from-theory-to-python-part-1/) model. It is responsible for computing the cell state and output at each timestep, and it consists of three gates: the input gate, the forget gate, and the output gate. Each of these gates is controlled by its own weight matrix.

Let's dive into the implementation of the LSTM Forward Cell in Python. Here is an example implementation:

```python
def lstm_cell_forward(input_t, previous_hidden_state, previous_memory_state, parameters):
    concat = np.concatenate((previous_hidden_state, input_t), axis=0)
    
    # compute values for each gate
    forget_gate = activation_sigmoid(np.dot(parameters['forget_gate_weights'].T, concat) + parameters['forget_gate_bias'])
    input_gate = activation_sigmoid(np.dot(parameters['input_gate_weights'].T, concat) + parameters['input_gate_bias'])
    input_gate_2 = activation_tanh(np.dot(parameters['gate_weights'].T, concat) + parameters['gate_bias'])
    output_gate = activation_sigmoid(np.dot(parameters['output_gate_weights'].T, concat) + parameters['output_gate_bias'])
    
    new_memory_state = np.multiply(forget_gate, previous_memory_state) + np.multiply(input_gate, input_gate_2)
    new_hidden_state = np.multiply(output_gate, activation_tanh(new_memory_state))
    
    # compute prediction of the LSTM cell
    yt_prediction = activation_softmax(np.dot(parameters['hidden_output_weights'].T, new_hidden_state) \
        + parameters['hidden_output_bias'])
    
    # store value for backpropagation
    cache = (new_hidden_state, new_memory_state, previous_hidden_state, previous_memory_state, \
        forget_gate, input_gate, input_gate_2, output_gate, parameters)
    
    return new_hidden_state, new_memory_state, yt_prediction, cache
```

In this implementation, we first compute the `input, forget`, and `output` gates using the input $X$, the previous hidden state `previous_hidden_state`, and the bias terms $b_i$, $b_f$, and $b_o$. We then compute the cell state `new_memory_state` using the gates and the previous cell state `previous_memory_state`. Finally, we compute the hidden state using the output gate and the cell state.

The cell state `new_memory_state` has the same shape as the hidden state, which is `(num_units, 1)`. The shape of the cell state is determined by the number of LSTM units or hidden neurons, which is specified when the LSTM layer is defined.

## Forward loop for all time series data

To process a sequence of data with LSTM, we need to implement a forward loop that iterates through each time step and computes the hidden state and output for each time step. The forward loop takes as input the sequential data, which is a tensor of shape `(num_time_steps, num_inputs)`, and the LSTM parameters, which include the weight matrices and bias terms for each gate.

![lstm-forward.png](lstm-forward.png)

Here's an example implementation of the forward loop using Python and NumPy:

```python
def lstm_forward(X, num_outputs, parameters):
    # Retrieve dimensions from input data shape
    num_time_steps, num_inputs = X.shape

    # Initialize hidden state and memory state arrays
    h = np.zeros((num_time_steps, num_units))
    C = np.zeros((num_time_steps, num_units))

    # Initialize output array
    Y = np.zeros((num_time_steps, num_outputs))

    # Initialize hidden state with zeros
    h[-1] = np.zeros(num_units)

    # Loop through each time step
    for t in range(num_time_steps):
        # Compute new values for the hidden state and memory state
        h[t], C[t], y, cache = lstm_cell_forward(X[t], h[t-1], C[t-1], parameters)

        # Store output value
        Y[t] = y

    # Store values needed for backpropagation in cache
    cache = (X, Y, h, C, parameters)

    return Y, cache
```

In this implementation, we first retrieve the number of time steps and number of inputs from the input data shape. We then initialize the hidden state and memory state arrays with zeros and the output array with `np.zeros`. We also initialize the hidden state with zeros. `num_outputs` refers to the number of output units in the LSTM layer. It is the size of the output vector produced by the LSTM layer at each time step.

We then loop through each time step and compute the new hidden state, memory state, and output using the `lstm_cell_forward()` function. We store the output value in the output array.

Finally, we store the input data, output data, hidden state, memory state, and parameters in a cache array, which will be used later for backpropagation.

## LSTM Backward Propagation

Once the output of the LSTM has been computed, we can use backpropagation to update the weights of the LSTM and improve its performance. Backpropagation is a process that allows us to compute the gradient of the loss function with respect to the weights of the LSTM, which allows us to update the weights using gradient descent.

The basic idea behind backpropagation is to start at the output of the LSTM and work backwards to compute the gradient of the loss function with respect to the weights. This gradient is then used to update the weights of the LSTM.

![lstm-backward.png](lstm-backward.png)

> The backward propagation process for LSTMs is similar to that of traditional neural networks, but the complexity of the LSTM architecture requires a more involved derivation of the gradients.
{: .prompt-tip}

The LSTM cell has three gates: the input gate, the forget gate, and the output gate. Each gate has its own set of weights, which are updated during the backward propagation process. The update process involves computing the partial derivative of the loss function with respect to each weight, which is used to update the weights through gradient descent.

Here's an example of how to implement LSTM backward propagation using Python and NumPy:

```python
def lstm_cell_backward(dy, dnext_hidden_state, dnext_memory_state, cache):
    # Unpack cache values
    new_hidden_state, new_memory_state, previous_hidden_state, previous_memory_state, \\
    forget_gate, input_gate, input_gate_2, output_gate, parameters = cache

    # Compute derivatives of output gate, cell state, and hidden state
    doutput_gate = dy * activation_tanh(new_memory_state)
    dnew_memory_state = dy * output_gate * (1 - activation_tanh(new_memory_state) ** 2) + dnext_hidden_state * dnext_memory_state

    # Compute derivatives of input, forget, and output gates
    dinput_gate = dnew_memory_state * input_gate_2 * (1 - input_gate) * input_gate
    dforget_gate = dnew_memory_state * previous_memory_state * (1 - forget_gate) * forget_gate
    dinput_gate_2 = dnew_memory_state * input_gate * (1 - input_gate_2 ** 2)
    doutput_gate = dy * activation_tanh(new_memory_state) * (1 - output_gate) * output_gate

    # Compute derivative of the concatenated input vector
    dconcat = np.zeros_like(concat)
    dconcat[:hidden_units, :] = dforget_gate
    dconcat[hidden_units:(hidden_units+input_units), :] = dinput_gate_2
    dconcat[(hidden_units+input_units):, :] = dinput_gate
    dconcat = np.dot(parameters['gate_weights'], dconcat.T)

    # Compute derivatives of previous hidden state and memory state
    dprevious_hidden_state = dconcat[:hidden_units, :]
    dprevious_memory_state = forget_gate * dnew_memory_state + dnext_memory_state * forget_gate

    # Compute derivatives of weight matrices and biases
    parameters['forget_gate_weights'] += np.dot(dforget_gate, np.concatenate((previous_hidden_state, input_t), axis=0).T)
    parameters['forget_gate_bias'] += np.sum(dforget_gate, axis=1, keepdims=True)

    parameters['input_gate_weights'] += np.dot(dinput_gate, np.concatenate((previous_hidden_state, input_t), axis=0).T)
    parameters['input_gate_bias'] += np.sum(dinput_gate, axis=1, keepdims=True)

    parameters['output_gate_weights'] += np.dot(doutput_gate, np.concatenate((previous_hidden_state, input_t), axis=0).T)
    parameters['output_gate_bias'] += np.sum(doutput_gate, axis=1, keepdims=True)

    parameters['gate_weights'] += np.dot(dinput_gate_2, np.concatenate((previous_hidden_state, input_t), axis=0).T)
    parameters['gate_bias'] += np.sum(dinput_gate_2, axis=1, keepdims=True)

    # Return derivatives of input, previous hidden state, and previous memory state
    dx = dconcat[hidden_units:(hidden_units+input_units), :]
    dprevious_hidden_state = dconcat[:hidden_units, :]
    dprevious_memory_state = forget_gate * dnew_memory_state + dnext_memory_state * forget_gate

    return dx, dprevious_hidden_state, dprevious_memory_state, parameters
```

```python
def lstm_backward(dY, cache):
    # Unpack cache values
    X, Y, h, C, parameters = cache

    # Initialize gradients
    dX = np.zeros_like(X)
    dparameters = {}
    dprevious_hidden_state = np.zeros_like(h[0])
    dprevious_memory_state = np.zeros_like(C[0])

    # Loop backwards through time steps
    for t in reversed(range(len(Y))):
        # Compute gradients for current time step
        dy = dY[t].reshape(1, -1)
        dx, dprevious_hidden_state, dprevious_memory_state, gradients = \
						lstm_cell_backward(dy, dprevious_hidden_state, dprevious_memory_state, cache[t])

        # Add gradients to total gradients
        dX[t, :] += dx
        for key, value in gradients.items():
            if key in dparameters:
                dparameters[key] += value
            else:
                dparameters[key] = value

    # Return gradients
    return dX, dparameters
```

In this implementation, we first unpack the cache variables and initialize the gradients to zero. We then loop backwards through the time steps and compute the gradients for each LSTM cell using `lstm_cell_backward()`. We add the gradients to the total gradients and return the gradients.

## Example with LSTM from Scratch

Here is an example of how to use the `lstm_forward()` and `lstm_backward()` functions to train an LSTM network and make predictions on a time series dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```python
# Load data
data = pd.read_csv('data.csv')
data = data.values
```

In this example, we first load the time series data and split it into training and testing sets. We then normalize the data using z-score normalization.

```python
# Split data into training and testing sets
train_data = data[:800, :]
test_data = data[800:, :]

# Normalize data
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
test_data = (test_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
```

We define the hyperparameters for the LSTM network, including the number of inputs and outputs, the number of hidden units, the learning rate, and the number of epochs. We also initialize the weights using the `create_weight_matrix()` function.

```python
# Initialize weights
parameters = create_weight_matrix(num_inputs, num_units)
```

We then train the LSTM network using a forward pass, a backward pass, and weight updates. We print the loss every 10 epochs to monitor the training process.

Finally, we make predictions on the test data using the trained LSTM network and plot the results.

```python
# Define hyperparameters
num_inputs = 1
num_outputs = 1
num_units = 128
learning_rate = 0.01
num_epochs = 100

# Train LSTM network
for epoch in range(num_epochs):
    # Forward pass
    Y, cache = lstm_forward(train_data[:, np.newaxis], num_outputs, parameters)

    # Compute loss
    loss = np.mean((Y - train_data[num_inputs:, np.newaxis]) ** 2)

    # Backward pass
    dY = (Y - train_data[num_inputs:, np.newaxis]) / (train_data.shape[0] - num_inputs)
    dX, gradients = lstm_backward(dY, cache)

    # Update weights
    for key, value in gradients.items():
        parameters[key] -= learning_rate * value

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss}')

# Make predictions on test data
Y_test, _ = lstm_forward(test_data[:, np.newaxis], num_outputs, parameters)

# Plot results
plt.plot(train_data[num_inputs:, :], label='Training Data')
plt.plot(range(train_data.shape[0]-num_outputs, data.shape[0]), Y, label='Predictions on Training Data')
plt.plot(range(train_data.shape[0], data.shape[0]), test_data[num_inputs:, :], label='Test Data')
plt.plot(range(train_data.shape[0], data.shape[0]), Y_test, label='Predictions on Test Data')
plt.legend()
plt.show()
```

Overall, the example demonstrates how to use the `lstm_forward()` and `lstm_backward()` functions to train an LSTM network and make predictions on time series data. By understanding the underlying theory of LSTMs and their implementation in Python, you can more effectively apply these powerful models to a wide range of time series problems.

## Recommended resources for further learning

Here are some recommended resources for further learning about Long Short Term Memory (LSTM):

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah
- [Long Short-Term Memory Networks (LSTM)](https://www.tensorflow.org/guide/keras/rnn#lstm) in TensorFlow's Keras documentation
- [Recurrent Neural Networks Tutorial, Part 4 – Implementing a GRU/LSTM RNN with Python and Theano](https://machinelearningmastery.com/develop-gated-recurrent-neural-networks-gru-and-lstm-in-python/) by Jason Brownlee

Here are some recommended books for delving into deep learning from scratch:
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://amzn.to/3YZeOAk) by Aurélien Géron
- [Deep Learning from Scratch: Building with Python from First Principles](https://amzn.to/40gyjFQ) by Seth Weidman
- [Data Science from Scratch: First Principles with Python](https://amzn.to/40ep3T7) by Joel Grus, a research engineer at the Allen Institute for Artificial Intelligence

## Conclusion

In conclusion, understanding the theory behind LSTMs is crucial to effectively use them in practice. With an understanding of the LSTM Forward Cell and Backward Propagation, as well as the implementation of LSTMs in Python, you can begin to apply these powerful models to a wide range of time series problems. By fine-tuning the weights and biases of the LSTM, you can achieve accurate predictions and gain valuable insights into complex systems over time.

> **Give me some ideas**: [Understanding Long Short Term Memory (LSTM) from Theory to Python - Part 1](https://www.codewithlinh.com/posts/lstm-from-theory-to-python-part-1/)
{: .prompt-info}

## References

[1] [https://pub.towardsai.net/building-a-lstm-from-scratch-in-python-1dedd89de8fe](https://pub.towardsai.net/building-a-lstm-from-scratch-in-python-1dedd89de8fe)