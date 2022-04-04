import argparse
import os
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def reduce_dim(features):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features)
    return principalComponents

def plot(datasets_feat, datasets_labels, results_path, name):
    principalDf = pd.DataFrame(data=datasets_feat
                               , columns=['First Component', 'Second Component'])

    classes = np.unique(datasets_labels)
    y_pd = pd.DataFrame(data=datasets_labels, columns=['target'])
    finalDf = pd.concat([principalDf, y_pd], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('First Component', fontsize=15)
    ax.set_ylabel('Second Component', fontsize=15)
    ax.set_title('Raw Activities Features', fontsize=20)
    cmap = get_cmap(len(classes))
    targets = []
    colors = []
    for i in range(len(classes)):
        targets.append(classes[i])
        colors.append(cmap(i))
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'First Component']
                   , finalDf.loc[indicesToKeep, 'Second Component']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    # plt.show()
    plt.savefig(os.path.join(results_path, name + '_plot.png'))
    plt.clf()


def plot_all(args):
    # file_path = args.file_path
    # results_path = args.results_path
    file_path = "Z:/Codes/DA-healthy2patient/3-data/f20_t5_3ways_target[mhealth]_source[_wharf_wisdm_uschad_pamap2]_baseline1_ICU_FSL.npz"
    results_path = "Z:/Codes/DA-healthy2patient/2-residuals/results/visualization/"

    folder_name = os.path.split(file_path)[-1].split('.npz')[0]
    results_path = os.path.join(results_path, folder_name)
    os.makedirs(results_path, exist_ok=True)

    dataset = np.load(file_path, allow_pickle=True)

    datasets_feat = []
    datasets_labels = []
    source_datasets = list(dataset['Xy_train'][0].keys())
    for source in source_datasets:
        print(f"Reducing dimentionality of {source}")
        train_data_, train_labels = dataset['Xy_train'][0][source]
        train_data = []
        for x in train_data_:
            train_data.append(np.squeeze(x).flatten())

        train_labels = np.array(train_labels)
        datasets_feat.extend(reduce_dim(np.array(train_data)))
        datasets_labels.extend(train_labels)

    print(f"Reducing dimentionality of target")
    test_data = []
    for x in dataset['X_test']:
        test_data.append(np.squeeze(x).flatten())
    datasets_feat.extend(reduce_dim(test_data))
    datasets_labels.extend(dataset['y_test'])

    plot(datasets_feat, datasets_labels, results_path)


def plot_each_one():

    # file_path = args.file_path
    # results_path = args.results_path
    file_path = "Z:/Codes/DA-healthy2patient/3-data/f20_t5_3ways_target[mhealth]_source[_wharf_wisdm_uschad_pamap2]_baseline1_ICU_FSL.npz"
    results_path = "Z:/Codes/DA-healthy2patient/2-residuals/results/visualization/"

    folder_name = os.path.split(file_path)[-1].split('.npz')[0]
    results_path = os.path.join(results_path, folder_name)
    os.makedirs(results_path, exist_ok=True)

    dataset = np.load(file_path, allow_pickle=True)

    source_datasets = list(dataset['Xy_train'][0].keys())
    for source in source_datasets:
        print(f"Reducing dimentionality of {source}")
        train_data_, train_labels = dataset['Xy_train'][0][source]
        train_data = []
        for x in train_data_:
            train_data.append(np.squeeze(x).flatten())

        train_labels = np.array(train_labels)
        datasets_feat = reduce_dim(np.array(train_data))
        datasets_labels = train_labels
        plot(datasets_feat, datasets_labels, results_path, source)


    print(f"Reducing dimentionality of target")
    test_data = []
    for x in dataset['X_test']:
        test_data.append(np.squeeze(x).flatten())
    datasets_feat = reduce_dim(test_data)
    datasets_labels = dataset['y_test']

    plot(datasets_feat, datasets_labels, results_path, "target")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--results_path', type=str, default='/shared/sense/sensor_domination/results/nshot_experiments/')
    parser.add_argument('--file_path', type=str, default="results/")
    args = parser.parse_args()

    plot_each_one()
