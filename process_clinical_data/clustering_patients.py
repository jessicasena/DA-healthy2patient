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


def A(sample):
    feat = []
    for col in range(0,sample.shape[1]):
        average = np.average(sample[:,col])
        feat.append(average)

    return feat


def SD(sample):
    feat = []
    for col in range(0, sample.shape[1]):
        std = np.std(sample[:, col])
        feat.append(std)

    return feat


def AAD(sample):
    feat = []
    for col in range(0, sample.shape[1]):
        data = sample[:, col]
        add = np.mean(np.absolute(data - np.mean(data)))
        feat.append(add)

    return feat


def ARA(sample):
    #Average Resultant Acceleration[1]:
    # Average of the square roots of the sum of the values of each axis squared √(xi^2 + yi^2+ zi^2) over the ED
    feat = []
    sum_square = 0
    sample = np.power(sample, 2)
    for col in range(0, sample.shape[1]):
        sum_square = sum_square + sample[:, col]

    sample = np.sqrt(sum_square)
    average = np.average(sample)
    feat.append(average)
    return feat


def COR(sample):
    feat = []
    for axis_i in range(0, sample.shape[1]):
        for axis_j in range(axis_i+1, sample.shape[1]):
            cor = np.corrcoef(sample[:,axis_i], sample[:,axis_j])
            feat.append(cor[0][1])

    return np.array(feat, dtype=np.float32)


def MOV(sample):
    var_threshold = 0.001
    var = np.mean([np.var(sample[:, 0]), np.var(sample[:, 1]), np.var(sample[:, 2])])
    if var > var_threshold:
        return 1
    else:
        return 0


def VAR(sample):
    feat = []
    for col in range(0, sample.shape[1]):
        var = np.var(sample[:, col])
        feat.append(var)

    return np.array(feat, dtype=np.float32)



def feature_extraction(X):
    #Extracts the features, as mentioned by Catal et al. 2015
    # Average - A,
    # Standard Deviation - SD,
    # Average Absolute Difference - AAD,
    # Average Resultant Acceleration - ARA(1),
    # Time Between Peaks - TBP
    X_tmp = []
    for sample in X:
        features = A(sample)
        features = np.hstack((features, SD(sample)))
        features = np.hstack((features, AAD(sample)))
        features = np.hstack((features, ARA(sample)))
        features = np.hstack((features, VAR(sample)))
        features = np.hstack((features, MOV(sample)))
        X_tmp.append(features)

    X = np.array(X_tmp)
    return X

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


def print_by_class(n_class, finalDf, exp_name, y_new):
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
        ax.scatter(finalDf.loc[indicesToKeep, 'First Component']
                   , finalDf.loc[indicesToKeep, 'Second Component']
                   , color=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    # plt.show()
    plt.savefig(exp_name + '.png')

def print_one_cluster(n_class, finalDf, exp_name, y_patients_id, cluster_number):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('First Component', fontsize=15)
    ax.set_ylabel('Second Component', fontsize=15)
    ax.set_title('Umap projection for patiend accelerometer data', fontsize=20)
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
    data_input_file = "/home/jsenadesouza/DA-healthy2patient/results/pain/dataset/f10_t1800_painscore_patientid_acc_30minmargin_measurednowcol_30min_10hz_filtered_kfold.npz"
    #feature_file = data_input_file.split('.npz')[0] + '_features.npz'
    #     np.savez_compressed(data_input_file.split('.npz')[0] + '_features.npz', X=X, y=y, folds=folds)

    print("Extracting features")
    tmp = np.load(data_input_file, allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']
    X = np.transpose(np.squeeze(X), (0, 2, 1))
    y_label, y_id = [], []
    for yy in y:
        y_label.append(yy.split("_")[0])
        y_id.append(yy.split("_")[1])

    y = np.array(y_label)

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

    y_patient = []
    for yy in y:
        y_patient.append(yy.split('_')[0])

    fold = "/home/jsenadesouza/DA-healthy2patient/results/plot_clusters/"

    n_class = np.unique(y_patient).shape[0]
    y_class = pd.DataFrame(data=y_patient, columns=['label'])
    finalDf = pd.concat([principalDf, y_class], axis=1)
    exp_name = 'clusteredby_pain_' + datetime.now().strftime("%d-%m-%y_%H-%M-%S") + "_"
    out_folder = os.path.join(fold, exp_name)
    print_by_class(n_class, finalDf, out_folder, y_class)

    n_class = np.unique(labels).shape[0]
    y_cluster = pd.DataFrame(data=labels, columns=['label'])
    finalDf = pd.concat([principalDf, y_cluster], axis=1)
    exp_name = 'clusteredbyalgo_' + datetime.now().strftime("%d-%m-%y_%H-%M-%S") + "_"
    out_folder = os.path.join(fold, exp_name)
    print_by_cluster(n_class, finalDf, out_folder, y_cluster)



