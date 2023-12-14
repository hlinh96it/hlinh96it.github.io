import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost


def learning_rate_schedule(num_epochs, epoch, sample):
    return (num_epochs - epoch) / sample


def stochastic_gradient_descent(X, y, weight, bias, num_epochs=100, num_train_sample=30):
    training_size = X.shape[0]

    for epoch in range(1, num_epochs):
        train_sample_idx = np.random.randint(low=0, high=training_size, size=num_train_sample)
        train_sample_data = np.take(X, train_sample_idx, axis=0)
        train_sample_label = np.take(y, train_sample_idx, axis=0)

        weight_derivative = -(2 / training_size) * sum(
            train_sample_data * (train_sample_label - (weight * train_sample_data + bias)))
        bias_derivative = -(2 / training_size) * sum(train_sample_label - (weight * train_sample_data + bias))

        # calculate learning rate
        learning_rate = learning_rate_schedule(epoch)

        weight -= learning_rate * weight_derivative
        bias -= learning_rate * bias_derivative

    return weight, bias


def batch_gradient_descent(X, y, weight, bias, learning_rate=0.01, num_iterations=200):
    training_size = X.shape[0]

    for idx in range(num_iterations):
        weight_derivative = -(2 / training_size) * sum(X * (y - (weight * X + bias)))
        bias_derivative = -(2 / training_size) * sum(y - (weight * X + bias))

        weight -= learning_rate * weight_derivative
        bias -= learning_rate * bias_derivative

        loss = mean_squared_error(y, weight * X + bias)
        print(f'Loss at iteration {idx}: {loss}')

    return weight, bias


def mini_batch_gradient_descent(X, y, weight, bias, num_epochs=100, num_train_sample=30):
    training_size = X.shape[0]

    for epoch in range(1, num_epochs):
        train_sample_idx = np.random.randint(low=0, high=training_size, size=num_train_sample)
        train_sample_data = np.take(X, train_sample_idx, axis=0)
        train_sample_label = np.take(y, train_sample_idx, axis=0)

        weight_derivative = -(2 / training_size) * sum(train_sample_data * (train_sample_label - (weight * train_sample_data + bias)))
        bias_derivative = -(2 / training_size) * sum(train_sample_label - (weight * train_sample_data + bias))

        weight -= learning_rate * weight_derivative
        bias -= learning_rate * bias_derivative

    return weight, bias


X = np.random.randn(100, 1)
y = 3 + 4 * X + np.random.randn(100, 1)
weight = np.random.random()
bias = np.random.random()

learning_rate = 0.1
num_epochs = 200
# weight, bias = stochastic_gradient_descent(X, y, weight, bias, num_epochs)
# print(weight, bias)

