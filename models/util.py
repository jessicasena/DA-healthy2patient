import math
import sys
import os
from multiprocessing.dummy import Pool

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from utils.data import SensorDataset
from sklearn.metrics import roc_curve
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from time import time
import logging


def Find_Optimal_Cutoff(target, predicted):  # Youden index
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]


    return list(roc_t['threshold'])

def split_data(X, y, y_target, labels2idx, patient_splits, folder_idx, logger=None):
    # split samples based on patients k-fold cross validation
    test_index, val_index, train_index = [], [], []
    for patient in patient_splits[folder_idx]:
        test_index.extend(list(np.where(y[:, -1] == patient)[0]))
    for patient in patient_splits[folder_idx + 1]:
        val_index.extend(list(np.where(y[:, -1] == patient)[0]))
    train_index = np.setdiff1d(np.arange(y.shape[0]), np.concatenate([test_index, val_index]))

    train_data, train_labels = X[train_index].squeeze(), y_target[train_index].squeeze()
    test_data, test_labels = X[test_index].squeeze(), y_target[test_index].squeeze()
    val_data, val_labels = X[val_index].squeeze(), y_target[val_index].squeeze()

    train_labels = np.array([labels2idx[label] for label in train_labels])
    test_labels = np.array([labels2idx[label] for label in test_labels])
    val_labels = np.array([labels2idx[label] for label in val_labels])

    if logger:
        logger.info(f"Folder {folder_idx + 1}")
        logger.info(f"Train data: {get_class_distribution(np.unique(train_labels, return_counts=True))}")
        logger.info(f"Test data: {get_class_distribution(np.unique(test_labels, return_counts=True))}")
        logger.info(f"Val data: {get_class_distribution(np.unique(val_labels, return_counts=True))}")

    return train_data, train_labels, test_data, test_labels, val_data, val_labels


def get_loaders(batch_size, sample_start, train_data, train_labels, test_data, test_labels, val_data=None, val_labels=None, weighted_sampler=False):
    train_set = SensorDataset(train_data, train_labels, sample_start, dataaug=True)
    test_set = SensorDataset(test_data, test_labels, sample_start)
    if val_data is not None and val_labels is not None:
        val_set = SensorDataset(val_data, val_labels, sample_start)

    if weighted_sampler:
        class_sample_count = np.array(
            [len(np.where(train_labels == t)[0]) for t in np.arange(0, len(np.unique(train_labels)))])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  pin_memory=True, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)

    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True)
    if val_data is not None and val_labels is not None:
        val_loader = DataLoader(val_set, batch_size=1, pin_memory=True)

        return train_loader, test_loader, val_loader
    else:
        return train_loader, test_loader

def load_data(filepath, clin_variable_target):
    # Load data
    dataset = np.load(filepath, allow_pickle=True)
    X = dataset["X"]
    y = dataset["y"]
    y_col_names = list(dataset['y_col_names'])
    col_idx_target = y_col_names.index(clin_variable_target)

    X, y = clean(X, y, col_idx_target)
    y_target = y[:, col_idx_target]#.astype(float)

    # not using pain level 0
    # idxs = np.where(y_target != 0)[0]
    # X_new, y_new, y_target_new = X[idxs], y[idxs], y_target[idxs]
    #
    # y_classes = [0 if yy < 5 else 1 for yy in y_target_new]
    # y_classes = np.array(y_classes)

    return X, y, y_target, y_col_names


def magnitude(sample):
    mag_vector = []
    for s in sample:
        mag_vector.append(math.sqrt(sum([s[0] ** 2, s[1] ** 2, s[2] ** 2])))
    return mag_vector


def load_data_mag(filepath, clin_variable_target):
    # Load data
    dataset = np.load(filepath, allow_pickle=True)
    X = dataset["X"][:500,:,:]
    y = dataset["y"][:500]
    y_col_names = list(dataset['y_col_names'])
    col_idx_target = y_col_names.index(clin_variable_target)

    X, y = clean(X, y, col_idx_target)
    y_target = y[:, col_idx_target]

    X_trasp = np.transpose(np.squeeze(X), (0, 1, 2))
    print("Extracting Features")
    start = time()
    with Pool(200) as p:
        X_feat = p.map(magnitude, X_trasp)
    end = time()
    print(f"{end - start:.4} seconds passed.")
    X = np.array(X_feat)

    return X, y, y_target


def rescale(x):
    x = (x - np.min(x) / (np.max(x) - np.min(x))) - 0.5
    return x


def clean(X, y, col_target):
    if '-1' in np.unique(y[:, col_target]):
        idxs = np.argwhere(np.array(y[:, col_target]) != '-1')
        X = X[idxs]
        y = y[idxs]
    return np.squeeze(X), np.squeeze(y)


def set_logger(filename):
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logging
    logger = logging.getLogger()
    logger.addHandler(console)

    return logger


def get_class_distribution(class_info):
    names, quant = class_info
    str = ""
    for name, q in zip(names, quant):
        str += f"{name}: {q/sum(quant)*100:.2f}% "
    return str
