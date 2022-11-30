import os
import pandas as pd
import numpy as np
from matplotlib import image

prefix = "./archive/Alzheimers-ADNI/"
test = os.path.join(prefix, "test")
train = os.path.join(prefix, "train")

classes = [c for c in os.listdir(train)]

# Maps sample IDs to genetic data (as a dataframe)
def load_genetic(path):
    csv_dict = dict()
    for c in classes:
        class_path = os.path.join(path, c)
        folder = os.listdir(class_path)
        for f in folder:
            if f.endswith('csv'):
               csv_dict[f.split('.')[0]] = pd.read_csv(os.path.join(class_path, f))
               # TODO: Add preprocessing of genetic data and write back to a file sans labels so it can be read as a numpy vector

# Loads all datasets into numpy arrays
# TODO: Update to properly load genetic data after preprocessing
def load_data(path):
    x_img, x_genetic, y = [], [], []
    for c in range(len(classes)):
        class_path = os.path.join(path, classes[c])
        folder = os.listdir(class_path)
        n = len(folder)
        x_img_c = []
        y_c = []

        for i in range(len(folder)):
            f = folder[i]
            if f.endswith('csv'):
                continue
            
            s = '_'.join(f.split('_')[1:4])
            #pd.read_csv(os.path.join(class_path, s))
            img = image.imread(os.path.join(class_path, f))
           
            x_img_c.append(img)
            row = [0 for _ in range(len(classes))]
            row[c] = 1
            y_c.append(row)

        x_img.extend(x_img_c)
        y.extend(y_c)

    x_img = np.array(x_img)
    y = np.array(y)
    return x_img, y


# train_genetic = load_genetic(train) 
# test_genetic = load_genetic(test)

# x_train_img, y_train = load_data(train)
# x_test_img, y_test = load_data(test)

                
           
            
            

