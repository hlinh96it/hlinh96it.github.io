# %% Import libraries
import os
import numpy as np
import cv2


# %% Importing dataset
train_path, test_path = 'Train', 'Test'
labels = {f: [] for f in os.listdir(train_path) if f != '.DS_Store'}

for label, img_list in labels.items():
    current_label = os.path.join(train_path, label)
    
    for data_dir in os.listdir(current_label):
        try:
            img = cv2.imread(os.path.join(current_label, data_dir))
            img_list.append(img)
        except:
            continue
        

# %%
