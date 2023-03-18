import numpy as np
from activation_function import *

def create_weight_matrix(input_units, hidden_units):
    """
    input_units -- number of input units \\
    hidden_units -- number of hidden units \\
    
    parameters['forget_gate_weights'] -- Weight matrix of the forget gate, shape (input_units + hidden_units, hidden_units)
    parameters['forget_gate_bias']  -- Bias of the forget gate, shape (hidden_units, 1)
    parameters['input_gate_weights'] -- Weight matrix of the update gate, shape (hidden_units, hidden_units + input_units)
    parameters['input_gate_bias'] -- Bias of the update gate, shape (hidden_units, 1)
    parameters['gate_weights'] -- Weight matrix of the first "tanh", shape (hidden_units, hidden_units + input_units)
    parameters['gate_bias'] --  Bias of the first "tanh", shape (hidden_units, 1)
    parameters['output_gate_weights'] -- Weight matrix of the output gate, shape (hidden_units, hidden_units + input_units)
    parameters['output_gate_bias'] --  Bias of the output gate, shape (hidden_units, 1)
    parameters['hidden_output_weights'] -- Weight matrix relating the hidden-state to the output, (n_y, hidden_units)
    parameters['hidden_output_bias'] -- Bias relating the hidden-state to the output (n_y, 1)
    """
    
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

def lstm_forward(input_X, inital_hidden_state, initial_params):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell.
    input_X -- input data for every time-step, shape (n_x, m, T_x)
    """
    
    caches = []
    weight_hidden_state = initial_params['hidden_output_weights']
    batch_size, input_dim, input_time = np.shape(input_X)
    

if __name__ == "__main__":
    parameters = create_weight_matrix()
    