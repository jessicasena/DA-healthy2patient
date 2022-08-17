import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from scipy import stats as st
from torch.utils.data import DataLoader

from utils.data import SensorSequence
from utils.models import CNNLSTM
from utils.utils import train, test
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from tensorboardX import SummaryWriter
from time import time

def rescale(x):
    x = (x - np.min(x) / (np.max(x) - np.min(x))) - 0.5
    return x


def clean(X, y):
    if '-1' in np.unique(y):
        idxs = np.argwhere(np.array(y) != '-1')
        X = X[idxs]
        y = y[idxs]
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model baselines')
    parser.add_argument('-v', type=str, help='Clinical variable to classify')

    args = parser.parse_args()
    filepath = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset//dataset_demographics_poi.npz"
    clin_variable = args.v
    dataset = np.load(filepath, allow_pickle=True)
    X = dataset["X"]
    X = rescale(X)
    y = dataset["y"]

    X_char = dataset["X_char"]
    X_poi = np.expand_dims(X_char[:, -1], 1).astype(np.float32)
    y_col_names = list(dataset['y_col_names'])
    col_idx = y_col_names.index(clin_variable)
    y = y[:, col_idx]
    X, y = clean(X, y)
    n_classes = np.unique(y).shape[0]
    labels2idx = {k: idx for idx, k in enumerate(np.unique(y))}
    use_cuda = torch.cuda.is_available()
    num_epochs = 100
    step = num_epochs/5
    batch_size = 128

    exp_name = f"exp_{time()}"
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

    cum_acc, cum_f1, cum_recall, cum_conf_matrices = [], [], [], []

    skf = StratifiedKFold(n_splits=5)

    for folder_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        writer = SummaryWriter(
            f'/home/jsenadesouza/DA-healthy2patient/results/outcomes/tensorboard/cnnlstm/exp_name/run_{time()}')

        train_data, train_poi, train_labels = X[train_index].squeeze(), X_poi[train_index], y[train_index].squeeze()
        test_data, test_poi, test_labels = X[test_index].squeeze(), X_poi[test_index], y[test_index].squeeze()

        train_labels = np.array([labels2idx[label] for label in train_labels])
        test_labels = np.array([labels2idx[label] for label in test_labels])

        train_set = SensorSequence(train_data, train_labels)
        test_set = SensorSequence(test_data, test_labels)

        train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=True)

        model = CNNLSTM(n_classes)

        model = model.to(device)

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        #criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)


        train(model, train_loader, optimizer, criterion, device, scheduler, writer, epochs=num_epochs, use_cuda=use_cuda)

        metrics = test(model, test_loader, device, use_cuda=use_cuda)

        for k, v in metrics.items():
            if k != 'confusion_matrix':
                print('Fold {} {}: {:.04f}'.format(folder_idx + 1, k.capitalize(), v))

        cum_acc.append(metrics['accuracy'])
        cum_f1.append(metrics['f1-score'])
        cum_recall.append(metrics['recall'])

    ci_mean = st.t.interval(0.9, len(cum_acc) - 1, loc=np.mean(cum_acc), scale=st.sem(cum_acc))
    ci_f1 = st.t.interval(0.9, len(cum_f1) -1, loc=np.mean(cum_f1), scale=st.sem(cum_f1))
    ci_recall = st.t.interval(0.9, len(cum_recall) -1, loc=np.mean(cum_recall), scale=st.sem(cum_recall))

    print('accuracy: {:.2f} ± {:.2f}'.format(np.mean(cum_acc) * 100, abs(np.mean(cum_acc) - ci_mean[0]) * 100))
    print('f1-score: {:.2f} ± {:.2f}'.format(np.mean(cum_f1) * 100, abs(np.mean(cum_f1) - ci_f1[0]) * 100))
    print('recall: {:.2f} ± {:.2f}'.format(np.mean(cum_recall) * 100, abs(np.mean(cum_recall) - ci_recall[0]) * 100))
