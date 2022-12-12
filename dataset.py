import os
import pandas as pd
import numpy as np
import cv2

prefix = "../archive/Alzheimers-ADNI/"
test = os.path.join(prefix, "test")
train = os.path.join(prefix, "train")

classes = ['Final AD JPEG', 'Final CN JPEG', 'Final EMCI JPEG', 'Final MCI JPEG']
gene_prefix = "GeneticData"
train_gene = os.path.join(gene_prefix, "train/train_dimred/train_PCA_49.csv")
test_gene = os.path.join(gene_prefix, "test/test_dimred/test_PCA_49.csv")

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
def load_data(path, gene_path=None):
    df = pd.read_csv(gene_path, index_col=0)
    failed = 0
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
            try:
                x_genetic.append(list(df.loc['X' + s]))
                img = cv2.imread(os.path.join(class_path, f))
            
                x_img_c.append(img)
                
                row = [0 for _ in range(len(classes))]
                row[c] = 1
                y_c.append(row)
            except:
                failed += 1
                continue

        x_img.extend(x_img_c)
        y.extend(y_c)

    x_img = np.array(x_img)
    y = np.array(y)
    x_genetic = np.array(x_genetic)
    print(y.shape)
    if not gene_path:
        return x_img, y
    else:
        return x_img, x_genetic, y

                
           
def load_shuffled_data():
    np.random.seed(42)
    x_train_img, x_train_genetic, y_train = load_data(train, train_gene)
    x_test_img, x_test_genetic, y_test = load_data(test, test_gene)

    x_img = np.vstack((x_train_img, x_test_img))
    x_genetic = np.vstack((x_train_genetic, x_test_genetic))
    y = np.vstack((y_train, y_test))

    n = x_img.shape[0]
    idxes = np.arange(n)
    np.random.shuffle(idxes)
    x_img, x_genetic, y = x_img[idxes], x_genetic[idxes], y[idxes]

    i = int(n * 0.8)
    return (x_img[:i], x_genetic[:i], y[:i]), (x_img[i+1:], x_genetic[i+1:], y[i+1:])



            


