import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np

import torch
import torch.nn as nn
from scipy import stats as st
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.data import SensorDataset
from utils.models import MetaSenseModeladdData
from models.transformer_model import TimeSeriesTransformer
from utils.utils import train, test
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from tensorboardX import SummaryWriter
from time import time
import logging

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

    logging.info(f'Target = {y_col_names[col_target]}\n')
    logging.info("\nCLinical variables used:")
    for idx in clin_var_idx:
       logging.info(f"{y_col_names[idx]}")

    return clin_var.astype(np.float32)

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
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    return logging

if __name__ == '__main__':
    start = time()
    parser = argparse.ArgumentParser('Model baselines')
    parser.add_argument('-v', type=str, default="pain_score_class", help='Clinical variable to classify')
    args = parser.parse_args()

    exp_name = f"exp_transformer_{args.v}_{time()}_focalloss.log"
    out_folder = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/"
    logging = set_logger(os.path.join(out_folder, exp_name))

    filepath = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/dataset_IntelligentICU_PAIN_ADAPT_Xchar_poi.npz"
    clin_variable_target = args.v
    logging.info(f"Clinical variable: {clin_variable_target}")
    dataset = np.load(filepath, allow_pickle=True)
    X = dataset["X"]
    X = rescale(X)
    y = dataset["y"]

    y_col_names = list(dataset['y_col_names'])
    #out_file.write(y_col_names)

    logging.info(f'Total of samples: {y.shape[0]}\n')
    logging.info(f'Total of hours: {y.shape[0] * 30 / 60}\n')
    pain_samples = np.count_nonzero(np.char.find(np.array(y[:, -1]), "P") != -1)
    adapt_samples = np.count_nonzero(np.char.find(np.array(y[:, -1]), "I") != -1)
    logging.info(
        f"Intelligent ICU samples: {len(y) - pain_samples - adapt_samples}, PAIN samples: {pain_samples}, ADAPT samples: {adapt_samples}")
    logging.info(f'Total of patients: {len(np.unique(y[:, -1]))}\n')
    pain_patients = np.count_nonzero(np.char.find(np.unique(y[:, -1]), "P") != -1)
    adapt_patients = np.count_nonzero(np.char.find(np.unique(y[:, -1]), "I") != -1)
    logging.info(
        f'Intelligent ICU patients: {len(np.unique(y[:, -1])) - adapt_patients - pain_patients}, PAIN patients: {pain_patients}, ADAPT patients: {adapt_patients}\n')

    col_idx_target = y_col_names.index(clin_variable_target)
    X, y = clean(X, y, col_idx_target)
    y_target = y[:, col_idx_target]
    n_classes = np.unique(y_target).shape[0]
    # X_char = dataset["X_char"].astype(np.float32)
    # X_add = get_clinical_data(y, y_col_names, clin_variable_target)
    # X_add = np.concatenate([X_add, X_char], axis=1)
    use_additional_data = False

    labels2idx = {k: idx for idx, k in enumerate(np.unique(y_target))}
    use_cuda = torch.cuda.is_available()
    num_epochs = 100
    batch_size_train = 40
    batch_size_test = 16
    #step = num_epochs / 5



    device = torch.device('cuda:1') if use_cuda else torch.device('cpu')

    cum_acc, cum_f1, cum_recall, cum_conf_matrices, cum_precision = [], [], [], [], []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logging.info(str(np.unique(y_target, return_counts=True)) + "")
    for folder_idx, (train_index, test_index) in enumerate(skf.split(X, y_target)):

        writer = SummaryWriter(
            f'/home/jsenadesouza/DA-healthy2patient/results/outcomes/tensorboard/transformers/exp_name/run_{time()}')

        train_data, train_labels = X[train_index].squeeze(), y_target[train_index].squeeze()
        test_data, test_labels = X[test_index].squeeze(),  y_target[test_index].squeeze()
        add_data_train, add_data_test = None, None
        # if use_additional_data:
        #    logging.info("Using additional data. shape: ", X_add.shape)
        #     add_data_train = X_add[train_index]
        #     add_data_test = X_add[test_index]

        train_labels = np.array([labels2idx[label] for label in train_labels])
        test_labels = np.array([labels2idx[label] for label in test_labels])

        class_sample_count = np.array(
            [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_set = SensorDataset(train_data, add_data_train, train_labels)
        test_set = SensorDataset(test_data, add_data_test, test_labels)

        train_loader = DataLoader(train_set, batch_size=batch_size_train, pin_memory=True, sampler=sampler)

        test_loader = DataLoader(test_set, batch_size=batch_size_train, pin_memory=True)

        if use_additional_data:
            model = MetaSenseModeladdData(n_classes, add_data_train.shape[1])
        else:
            model = TimeSeriesTransformer(n_classes, batch_first=False,
                                      dim_val=512,
                                      n_encoder_layers= 6,
                                      n_heads = 8,
                                      dropout_encoder = 0.1,
                                      dropout_pos_enc = 0.1,
                                      dim_feedforward_encoder= 128)

        model = model.to(device)

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-04)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

        train(model, train_loader, optim, criterion, device, scheduler, writer, epochs=num_epochs, use_cuda=use_cuda, use_additional_data=use_additional_data)

        metrics = test(model, test_loader, device, use_cuda=use_cuda, use_additional_data=use_additional_data)

        for k, v in metrics.items():
            if k != 'confusion_matrix':
                logging.info('Fold {} {}: {}\n'.format(folder_idx + 1, k.capitalize(), v))

        cum_acc.append(metrics['accuracy'])
        cum_f1.append(metrics['f1-score'])
        cum_recall.append(metrics['recall'])
        cum_precision.append(metrics['precision'])

    for class_ in range(len(np.unique(y_target))):
        logging.info(f"Class: {class_}")
        current_acc = np.array(cum_acc)
        current_f1 = np.array(cum_f1)[:, class_]
        current_recall = np.array(cum_recall)[:, class_]
        current_prec = np.array(cum_precision)[:, class_]
        ci_mean = st.t.interval(0.95, len(current_acc) - 1, loc=np.mean(current_acc), scale=st.sem(current_acc))
        ci_f1 = st.t.interval(0.95, len(current_f1) -1, loc=np.mean(current_f1), scale=st.sem(current_f1))
        ci_recall = st.t.interval(0.95, len(current_recall) -1, loc=np.mean(current_recall), scale=st.sem(current_recall))
        #ci_auc = st.t.interval(0.95, len(cum_auc) -1, loc=np.mean(cum_auc), scale=st.sem(cum_auc))
        # ci_AUROC = st.t.interval(0.95, len(cum_AUROC) -1, loc=np.mean(cum_AUROC), scale=st.sem(cum_AUROC))
        ci_prec = st.t.interval(0.95, len(current_prec) -1, loc=np.mean(current_prec), scale=st.sem(current_prec))

        logging.info('accuracy: {:.2f} ± {:.2f}\n'.format(np.mean(current_acc) * 100, abs(np.mean(current_acc) - ci_mean[0]) * 100))
        logging.info('recall: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall) * 100, abs(np.mean(current_recall) - ci_recall[0]) * 100))
        logging.info('f1-score: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1) * 100, abs(np.mean(current_f1) - ci_f1[0]) * 100))
        logging.info('precision: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec) * 100, abs(np.mean(current_prec) - ci_prec[0]) * 100))

    end = time()
    logging.info('Epochs: {} Time: {:.2f} seconds\n'.format(num_epochs, (end - start)))
