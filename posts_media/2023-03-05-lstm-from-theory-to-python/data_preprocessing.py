import numpy as np

def map_character_to_index(vocab):
    char_id, id_char = {}, {}
    for ii, char in enumerate(vocab):
        char_id[char] = ii
        id_char[ii] = char
        
    return char_id, id_char

def make_custom_dataset(transform_data, batch_size, vocab, char_id):
    train_dataset = []

    # split the transformed data into batches of batch_size
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
        
    return train_dataset