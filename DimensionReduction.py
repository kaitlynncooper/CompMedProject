import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA
import plotly.express as px

pc2_columns = ['principal component 1', 'principal component 2']
pc3_columns = ['principal component 1', 'principal component 2', 'principal component 3']

def linear_pca(dataframe, path):
    features = dataframe.columns.tolist()[:-1]
    x = dataframe.loc[:, features].values
    y = dataframe.loc[:,['Label']].values
    x = StandardScaler().fit_transform(x)

    pca_2 = PCA(n_components = 2)
    principalComponents_2 = pca_2.fit_transform(x)
    total_var = pca_2.explained_variance_ratio_.sum() * 100
    plot_and_save(principalComponents_2, pc2_columns, dataframe.index, 2, total_var, dataframe['Label'], path+"_pca_2")

    pca_3 = PCA(n_components = 3)
    principalComponents_3 = pca_3.fit_transform(x)
    total_var = pca_3.explained_variance_ratio_.sum() * 100
    plot_and_save(principalComponents_3, pc3_columns, dataframe.index, 3, total_var, dataframe['Label'], path+"_pca_3")
    
    return

def calculate_kernel_expvar(kpca):
    explained_var = np.var(kpca, axis = 0)
    return explained_var.sum() * 100

def kernel_pca(dataframe, path):
    features = dataframe.columns.tolist()[:-1]
    x = dataframe.loc[:, features].values
    y = dataframe.loc[:,['Label']].values
    x = StandardScaler().fit_transform(x)

    kernel_pca_2 = KernelPCA(n_components = 2, kernel = 'rbf')
    kernel_pca_components_2 = kernel_pca_2.fit_transform(x)
    kernel_pca_var = calculate_kernel_expvar(kernel_pca_components_2)
    plot_and_save(kernel_pca_components_2, pc2_columns, dataframe.index, 2, kernel_pca_var, dataframe['Label'], path+"_kernelpca2")

    kernel_pca_3 = KernelPCA(n_components = 3, kernel = 'rbf')
    kernel_pca_components_3 = kernel_pca_3.fit_transform(x)
    kernel_pca_var = calculate_kernel_expvar(kernel_pca_components_3)
    plot_and_save(kernel_pca_components_3, pc3_columns, dataframe.index, 3, kernel_pca_var, dataframe['Label'], path+"_kernelpca3")

    return


def ICA(dataframe, path):
    features = dataframe.columns.tolist()[:-1]
    x = dataframe.loc[:, features].values
    y = dataframe.loc[:,['Label']].values
    x = StandardScaler().fit_transform(x)

    fast_ICA_2 = FastICA(n_components = 2)
    fast_ICA_components_2 = fast_ICA_2.fit_transform(x)
    plot_and_save(fast_ICA_components_2, pc2_columns, dataframe.index, 2, 0, dataframe['Label'], path+"_ICA2")

    fast_ICA_3 = FastICA(n_components = 3)
    fast_ICA_components_3 = fast_ICA_3.fit_transform(x)
    plot_and_save(fast_ICA_components_3, pc3_columns, dataframe.index, 3, 0, dataframe['Label'], path+"_ICA3")

    return


def plot_and_save(components, columns, rows, ndim, variance, labels, path):
    dataframe = pd.DataFrame(data = components, columns = columns)
    dataframe.index = rows
    dataframe.to_csv(path+".csv")
    if ndim == 2:
        fig = px.scatter(components, x = 0, y = 1, color = labels, title = f'Total Explained Variance: {variance: .2f}%')
        fig.write_image(path+".jpeg")
        #fig.show()
    if ndim == 3:
        fig = px.scatter_3d(components, x = 0, y = 1, z = 2, color = labels, title = f'Total Explained Variance: {variance: .2f}%', labels = {'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'})
        fig.write_image(path+".jpeg")
        #fig.show()
    return

def create_df(dataframe, metadata):
    dataframe = dataframe.transpose()
    dataframe.sort_index(inplace = True)

    metadata = metadata.sort_values(metadata.columns[0], ascending = True)

    temp_df = pd.DataFrame()
    temp_df['fillerlabel'] = ['Label']
    metadata_labels = pd.concat([temp_df, metadata['Label']]).reset_index(drop = True)
    metadata_labels.columns = ['fillerlabel', 'Label']
    metadata_labels = metadata_labels['Label'].fillna(metadata_labels['fillerlabel'])

    dataframe_complete = dataframe
    dataframe_complete['Label'] = metadata_labels.values

    column_labels = dataframe_complete.iloc[0]
    dataframe_complete = dataframe_complete[1:]
    dataframe_complete.columns = column_labels

    return dataframe_complete


if __name__ == "__main__":
    print("Started...")
    train_data = pd.read_csv("Geneticdata/train/train_filtered.csv")
    train_metadata = pd.read_csv("Geneticdata/train/train_metadata_filtered.csv")
    test_data = pd.read_csv("Geneticdata/test/test_filtered.csv")
    test_metadata = pd.read_csv("Geneticdata/test/test_metadata_filtered.csv")
    
    train_path = "Geneticdata/train/train_dimred/train"
    test_path = "Geneticdata/test/test_dimred/test"

    print("Creating dataframes...")
    train_df = create_df(train_data, train_metadata)
    test_df = create_df(test_data, test_metadata)
    
    print("Performing linear PCA...")
    linear_pca(train_df, train_path)
    linear_pca(test_df, test_path)

    print("Performing kernel PCA...")
    kernel_pca(train_df, train_path)
    kernel_pca(test_df, test_path)

    print("Performing ICA...")
    ICA(train_df, train_path)
    ICA(test_df, test_path)