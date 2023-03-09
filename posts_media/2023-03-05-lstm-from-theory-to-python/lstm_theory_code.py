import pandas as pd
import numpy as np


def make_equal_length(name, max_length):
    return name + '.'*(max_length - len(name))

def activation_sigmoid(X):
    return 1 / (1 + np.exp(-X))

def activation_tanh(X):
    return np.tanh(X)

def activation_softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)

def tanh_derivative(X):
    return 1 - (X**2)

def initial_parameters(input_units, hidden_units):
    mean, std = 0, 0.01
    
    # lstm cell weights
    forget_gate_weights = np.random.normal(loc=mean, scale=std,
                                           size=(input_units + hidden_units, hidden_units))


#%% Data preprocessing   ===========================================================================
# read data, lower case, and reshape to numpy array
data = pd.read_csv('NationalNames.csv')['Name'][:2000].apply(str.lower).to_numpy().reshape(-1, 1)

# transform names to equal length by adding .
max_length_name = np.vectorize(len)(data).max()
transform_data = np.vectorize(make_equal_length)(data, max_length_name)

# create a list of all possible characters
vocab = set(''.join(transform_data.flatten()))

# map characters to idx and vice versa
char_id, id_char = {}, {}
for ii, char in enumerate(vocab):
    char_id[char] = ii
    id_char[ii] = char
    

#%% Make train dataset  ===========================================================================
train_dataset = []
batch_size = 20

# split the transformed data into batches of 20
for i in range(len(transform_data)-batch_size+1):
    start = i*batch_size
    end = start+batch_size
    
    #batch data
    batch_data = transform_data[start:end]
    
    if(len(batch_data)!=batch_size):
        break
        
    #convert each char of each name of batch data into one hot encoding
    char_list = []
    for k in range(len(batch_data[0][0])):
        batch_dataset = np.zeros([batch_size, len(vocab)])
        for j in range(batch_size):
            name = batch_data[j][0]
            char_index = char_id[name[k]]
            batch_dataset[j, char_index] = 1.0
     
        #store the ith char's one hot representation of each name in batch_data
        char_list.append(batch_dataset)
    
    #store each char's of every name in batch dataset into train_dataset
    train_dataset.append(char_list)
    
    
#%% Config hyperparameters   ===========================================================================
input_units, hidden_units, output_units = 100, 256, max_length_name
learning_rate, beta1, beta2 = 0.005, 0.90, 0.99


#%% Initialize parameters   ===========================================================================
parameters = initialize_parameters()