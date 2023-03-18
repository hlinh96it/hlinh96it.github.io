import pandas as pd
import numpy as np

from lstm_module import *
from activation_function import *
from data_preprocessing import *


def make_equal_length(name, max_length):
    return name + '.'*(max_length - len(name))


#%% Data preprocessing   ===========================================================================
# read data, lower case, and reshape to numpy array
data = pd.read_csv('NationalNames.csv')['Name'][:2000].apply(str.lower).to_numpy().reshape(-1, 1)

# transform names to equal length by adding .
max_length_name = np.vectorize(len)(data).max()
transform_data = np.vectorize(make_equal_length)(data, max_length_name)

vocab = set(''.join(transform_data.flatten()))  # create a list of all possible characters
char_id, id_char = map_character_to_index(vocab)  # map characters to idx and vice versa
    

#%% Make train dataset  ===========================================================================
train_dataset = make_custom_dataset(transform_data, batch_size=20, vocab=vocab, char_id=char_id)
    
    
#%% Config hyperparameters   ===========================================================================
input_units, hidden_units, output_units = 100, 256, max_length_name
learning_rate, beta1, beta2 = 0.005, 0.90, 0.99


#%% Initialize parameters   ===========================================================================
parameters = initial_parameters(input_units, hidden_units)