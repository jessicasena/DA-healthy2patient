import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np

import torch
import torch.nn as nn
from scipy import stats as st
from torch.utils.data import DataLoader

from utils.data import SensorDataset, SensorPublicDataset
from utils.models import MetaSenseModel, MetaSenseModeladdData
from utils.transformer_model import TimeSeriesTransformer
from utils.utils import train, test
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from tensorboardX import SummaryWriter
from time import time


def rescale(x):
    x = (x - np.min(x) / (np.max(x) - np.min(x))) - 0.5
    return x


def clean(X, y, col_target):
    if '-1' in np.unique(y[:, col_target]):
        idxs = np.argwhere(np.array(y[:, col_target]) != '-1')
        X = X[idxs]
        y = y[idxs]
    return np.squeeze(X), np.squeeze(y)


def get_clinical_data(y, y_col_names, target_col_name):
    regression_val = [0, 2, 6, 8, 10, 14, 16]
    col_target = y_col_names.index(target_col_name)
    col_target_reg = y_col_names.index(target_col_name.split("_class")[0])

    clin_var_idx = []
    for ii in regression_val:
        ii = int(ii)
        if ii != col_target and ii != col_target_reg:
            clin_var_idx.append(ii)

    clin_var = y[:, clin_var_idx]

    print(f'Target = {y_col_names[col_target]}')
    print("\nCLinical variables used:\n")
    for idx in clin_var_idx:
        print(f"{y_col_names[idx]}")

    return clin_var.astype(np.float32)


if __name__ == '__main__':

    filepath = "/home/jsenadesouza/DA-healthy2patient/data/public/WISDM.npz"
    dataset = np.load(filepath, allow_pickle=True)
    X = dataset["X"]
    y = dataset["y"]

    y_one_hot = y
    y_categoric = y.argmax(axis=1)
    n_classes = np.unique(y_categoric).shape[0]

    use_cuda = torch.cuda.is_available()
    num_epochs = 50
    batch_size_train = 256
    batch_size_test = 256
    step = num_epochs / 5

    exp_name = f"exp_{time()}"

    device = torch.device('cuda:2') if use_cuda else torch.device('cpu')

    cum_acc, cum_f1, cum_recall, cum_conf_matrices = [], [], [], []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for folder_idx, (train_index, test_index) in enumerate(skf.split(X, y_categoric)):

        writer = SummaryWriter(
            f'/home/jsenadesouza/DA-healthy2patient/results/outcomes/tensorboard/cnnlstm/exp_name/run_{time()}')

        train_data, train_labels = X[train_index].squeeze(), y_categoric[train_index].squeeze()
        test_data, test_labels = X[test_index].squeeze(),  y_categoric[test_index].squeeze()
        add_data_train, add_data_test = None, None

        train_set = SensorPublicDataset(train_data, add_data_train, train_labels)
        test_set = SensorPublicDataset(test_data, add_data_test, test_labels)

        train_loader = DataLoader(train_set, batch_size=batch_size_train, pin_memory=True, shuffle=True)

        test_loader = DataLoader(test_set, batch_size=batch_size_train, pin_memory=True)

        model = TimeSeriesTransformer(n_classes, batch_first=True,
                                      dim_val=32,
                                      n_encoder_layers= 4,
                                      n_heads = 8,
                                      dropout_encoder = 0.2,
                                      dropout_pos_enc = 0.1,
                                      dim_feedforward_encoder= 2048)

        model = model.to(device)

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        # criterion = nn.BCELoss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step, gamma=0.5)

        train(model, train_loader, optim, criterion, device, scheduler, writer, epochs=num_epochs, use_cuda=use_cuda, use_additional_data=False)

        metrics = test(model, test_loader, device, use_cuda=use_cuda, use_additional_data=False)

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
