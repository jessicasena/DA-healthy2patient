import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import umap
import copy
import hdbscan
from imblearn.under_sampling import RandomUnderSampler
from multiprocessing import Pool
import os
from datetime import datetime
import math

def magnitude(sample):
    mag_vector = []
    for s in sample:
        mag_vector.append(math.sqrt(sum([s[0]**2, s[1]**2, s[2]**2])))
    return mag_vector


def feature_extraction(X):
    """
    Derive three activity intensity cues: mean and standard deviation of activity intensity,
    and duration of immobility during assessment window to summarize the data.
    """
    features = []
    for sample in X:
        mag = magnitude(sample)
        ft_mean = np.mean(mag)
        ft_std = np.std(mag)
        # calculate duration of immobility
        features.append([ft_mean, ft_std])

    return features


def get_cmap(n, name='gist_rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def print_by_cluster(n_class, finalDf, exp_name, y_new):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('First Component', fontsize=15)
    ax.set_ylabel('Second Component', fontsize=15)
    ax.set_title('Umap projection for patiend accelerometer data', fontsize=20)
    cmap = get_cmap(n_class)
    targets = np.unique(y_new)
    colors = []
    for i in range(n_class):
        colors.append(cmap(i))

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['label'] == target
        if target <= 0:
            ax.scatter(finalDf.loc[indicesToKeep, 'First Component']
                       , finalDf.loc[indicesToKeep, 'Second Component']
                       , color='grey'
                       , s=50)
        else:
            ax.scatter(finalDf.loc[indicesToKeep, 'First Component']
                       , finalDf.loc[indicesToKeep, 'Second Component']
                       , color=color
                       , s=50)
    ax.legend(targets)
    ax.grid()
    # plt.show()
    plt.savefig(exp_name + '.png')


def print_by_class(n_class, finalDf, exp_name, y_new, clin_variable):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('First Component', fontsize=15)
    ax.set_ylabel('Second Component', fontsize=15)
    ax.set_title(f'Accelerometer data - {clin_variable}', fontsize=20)
    cmap = get_cmap(n_class)
    targets = np.unique(y_new)
    colors = []
    for i in range(n_class):
        colors.append(cmap(i))

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'First Component']
                   , finalDf.loc[indicesToKeep, 'Second Component']
                   , color=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    # plt.show()
    plt.savefig(exp_name + '.png')


def print_one_cluster(n_class, finalDf, exp_name, y_patients_id, cluster_number, clin_variable):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('First Component', fontsize=15)
    ax.set_ylabel('Second Component', fontsize=15)
    ax.set_title(f'Accelerometer data - {clin_variable}', fontsize=20)
    cmap = get_cmap(n_class)
    targets = np.unique(y_patients_id)
    colors = []
    for i in range(n_class):
        colors.append(cmap(i))

    finalDf = finalDf.loc[finalDf['target'] == cluster_number]

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'First Component']
                       , finalDf.loc[indicesToKeep, 'Second Component']
                       , color=color
                       , s=50)

    ax.legend(targets)
    ax.grid()
    # plt.show()
    plt.savefig(exp_name + '.png')


if __name__ == '__main__':
    np.random.seed(42)
    data_input_file = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/f10_t1800_outcomesscore_patientid_acc_30minmargin_measurednowcol_30min_10hz_filtered.npz"
    #feature_file = data_input_file.split('.npz')[0] + '_features.npz'
    #     np.savez_compressed(data_input_file.split('.npz')[0] + '_features.npz', X=X, y=y, folds=folds)

    print("Extracting features")
    tmp = np.load(data_input_file, allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    y_col_names = list(tmp['y_col_names'])
    X = np.transpose(np.squeeze(X), (0, 2, 1))

    # Undersampling the data to balance the classes
    # rus = RandomUnderSampler(random_state=42, sampling_strategy={'none': 1000, 'mild': 1000, 'moderate': 1000, 'severe': 1000})
    # rus.fit_resample(X[:, :, 0], y)
    # X, y = X[rus.sample_indices_], y[rus.sample_indices_]
    # print("Undersampling done")

    X_feat = feature_extraction(X)

    embedding = umap.UMAP(
        n_neighbors=10,
        min_dist=0.0,
        n_components=2,
        random_state=42#,
        #metric='cosine'
    ).fit_transform(X_feat)

    principalDf = pd.DataFrame(data=embedding
                               , columns=['First Component', 'Second Component'])

    labels = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=500,
    ).fit_predict(embedding)

    fold = "/home/jsenadesouza/DA-healthy2patient/results/plot_clusters/"

    for clin_variable in y_col_names:
        #clin_variable = 'patient_id'
        col_idx = y_col_names.index(clin_variable)
        y_target = np.array(y[:, col_idx])

        n_class = np.unique(y_target).shape[0]
        y_class = pd.DataFrame(data=y_target, columns=['label'])
        finalDf = pd.concat([principalDf, y_class], axis=1)
        exp_name = f'clusteredby_{clin_variable}_' + datetime.now().strftime("%d-%m-%y_%H-%M-%S") + "_"
        out_folder = os.path.join(fold, exp_name)
        print_by_class(n_class, finalDf, out_folder, y_class, clin_variable)



