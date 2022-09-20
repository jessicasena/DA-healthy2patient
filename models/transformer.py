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
from models.transformer_model import TimeSeriesTransformer
from utils.utils import train, test
from models.transformer_model import FocalLoss
from utils.data import DA_Jitter, DA_Scaling, DA_TimeWarp, DA_MagWarp, DA_Permutation, DA_Rotation, DA_RandSampling

from tensorboardX import SummaryWriter
from time import time
import torchvision
import logging
import random
from tqdm import tqdm


class Transformer:
    def __init__(self, data_path, exp_name, out_folder, num_epochs, batch_size, data_aug=False, loss='cross_entropy',
                 weighted_sampler=False):

        self.data_path = data_path
        self.exp_name = exp_name
        self.out_folder = out_folder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.loss = loss
        self.weighted_sampler = weighted_sampler
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
        self.X, self.y, self.y_target = load_data(self.data_path)        
        self.n_classes = np.unique(self.y_target).shape[0]
        self.labels2idx = {k: idx for idx, k in enumerate(np.unique(self.y_target))}
        set_logger(os.path.join(self.out_folder, self.exp_name))

    def print_info(self):
        logging.info(f"Experiment name: {self.exp_name}")
        logging.info(f'Total of samples: {self.y.shape[0]}\n')
        logging.info(f'Total of hours: {self.y.shape[0] * 30 / 60}\n')
        pain_samples = np.count_nonzero(np.char.find(np.array(self.y[:, -1]), "P") != -1)
        adapt_samples = np.count_nonzero(np.char.find(np.array(self.y[:, -1]), "I") != -1)
        logging.info(
            f"Intelligent ICU samples: {len(self.y) - pain_samples - adapt_samples}, PAIN samples: {pain_samples}, "
            f"ADAPT samples: {adapt_samples}")
        logging.info(f'Total of patients: {len(np.unique(self.y[:, -1]))}\n')
        pain_patients = np.count_nonzero(np.char.find(np.unique(self.y[:, -1]), "P") != -1)
        adapt_patients = np.count_nonzero(np.char.find(np.unique(self.y[:, -1]), "I") != -1)
        logging.info(
            f'Intelligent ICU patients: {len(np.unique(self.y[:, -1])) - adapt_patients - pain_patients}, '
            f'PAIN patients: {pain_patients}, ADAPT patients: {adapt_patients}\n')
        logging.info(str(np.unique(self.y_target, return_counts=True)) + "")

        # print variable info
        logging.info(f"Data augmentation: {self.data_aug}")
        logging.info(f"Loss: {self.loss}")
        logging.info(f"Weighted sampler: {self.weighted_sampler}")

    def print_metrics(self, cum_acc, cum_recall, cum_precision, cum_auc, cum_f1):
        for class_ in range(len(np.unique(self.y_target))):
            logging.info(f"Class: {class_}")
            current_acc = np.array(cum_acc)
            current_f1 = np.array(cum_f1)[:, class_]
            current_recall = np.array(cum_recall)[:, class_]
            current_prec = np.array(cum_precision)[:, class_]
            current_auc = np.array(cum_auc)
            ci_mean = st.t.interval(0.95, len(current_acc) - 1, loc=np.mean(current_acc), scale=st.sem(current_acc))
            ci_f1 = st.t.interval(0.95, len(current_f1) - 1, loc=np.mean(current_f1), scale=st.sem(current_f1))
            ci_recall = st.t.interval(0.95, len(current_recall) - 1, loc=np.mean(current_recall),
                                      scale=st.sem(current_recall))
            ci_prec = st.t.interval(0.95, len(current_prec) - 1, loc=np.mean(current_prec), scale=st.sem(current_prec))
            ci_auc = st.t.interval(0.95, len(current_auc) - 1, loc=np.mean(current_auc), scale=st.sem(current_auc))

            logging.info('accuracy: {:.2f} ± {:.2f}\n'.format(np.mean(current_acc) * 100,
                                                              abs(np.mean(current_acc) - ci_mean[0]) * 100))
            logging.info('recall: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall) * 100,
                                                            abs(np.mean(current_recall) - ci_recall[0]) * 100))
            logging.info('precision: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec) * 100,
                                                               abs(np.mean(current_prec) - ci_prec[0]) * 100))
            logging.info('f1-score: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1) * 100,
                                                              abs(np.mean(current_f1) - ci_f1[0]) * 100))
            logging.info('roc_auc: {:.2f} ± {:.2f}\n'.format(np.mean(current_auc) * 100,
                                                             abs(np.mean(current_auc) - ci_auc[0]) * 100))

    def split_data(self, patient_splits, folder_idx):
        # split samples based on patients k-fold cross validation
        test_index, val_index, train_index = [], [], []
        for patient in patient_splits[folder_idx]:
            test_index.extend(list(np.where(self.y[:, -1] == patient)[0]))
        for patient in patient_splits[folder_idx + 1]:
            val_index.extend(list(np.where(self.y[:, -1] == patient)[0]))
        train_index = np.setdiff1d(np.arange(self.y.shape[0]), np.concatenate([test_index, val_index]))

        train_data, train_labels = self.X[train_index].squeeze(), self.y_target[train_index].squeeze()
        test_data, test_labels = self.X[test_index].squeeze(), self.y_target[test_index].squeeze()
        val_data, val_labels = self.X[val_index].squeeze(), self.y_target[val_index].squeeze()

        train_labels = np.array([self.labels2idx[label] for label in train_labels])
        test_labels = np.array([self.labels2idx[label] for label in test_labels])
        val_labels = np.array([self.labels2idx[label] for label in val_labels])

        logging.info(f"Folder {folder_idx + 1}")
        logging.info(f"Train data: {get_class_distribution(np.unique(train_labels, return_counts=True))}")
        logging.info(f"Test data: {get_class_distribution(np.unique(test_labels, return_counts=True))}")
        logging.info(f"Val data: {get_class_distribution(np.unique(val_labels, return_counts=True))}")

        return train_data, train_labels, test_data, test_labels, val_data, val_labels

    def get_loaders(self, train_data, train_labels, test_data, test_labels, val_data, val_labels):
        train_set = SensorDataset(train_data, None, train_labels, dataaug=self.data_aug)
        test_set = SensorDataset(test_data, None, test_labels)
        val_set = SensorDataset(val_data, None, val_labels)

        if self.weighted_sampler:
            class_sample_count = np.array(
                [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in train_labels])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                      pin_memory=True, sampler=sampler)
        else:
            train_loader = DataLoader(train_set, batch_size=self.batch_size, pin_memory=True)

        test_loader = DataLoader(test_set, batch_size=self.batch_size, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, pin_memory=True)

        return train_loader, test_loader, val_loader

    def run(self):
        cum_acc, cum_recall, cum_precision, cum_auc, cum_f1 = [], [], [], [], []
        self.print_info()

        # Split data
        patients = list(np.unique(self.y[:, -1]))
        random.shuffle(patients)
        patient_splits = np.array_split(patients, 6)

        for folder_idx in range(5):
            # writer = SummaryWriter(
            #     f'/home/jsenadesouza/DA-healthy2patient/results/outcomes/tensorboard/transformers/self.exp_name/run_{time()}')

            # split the data into train, val and test sets
            train_data, train_labels, test_data, test_labels, val_data, val_labels = self.split_data(patient_splits,
                                                                                                     folder_idx)
            # get dataloaders
            train_loader, test_loader, val_loader = self.get_loaders(train_data, train_labels, test_data, test_labels,
                                                                     val_data, val_labels)

            model = create_model()
            model = model.to(self.device)
            if self.loss == "focal":
                criterion = FocalLoss(alpha=0.9, gamma=3)
            else:
                criterion = nn.BCEWithLogitsLoss(reduction='mean')
            optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-04)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
            earling_stopping = {"val_loader": test_loader, "last_loss": 100, "patience": 10, "triggertimes": 0}
            best_model_folder = os.path.join(self.out_folder, "best_model")
            if not os.path.exists(best_model_folder):
                os.makedirs(best_model_folder)
            best_model_path = os.path.join(best_model_folder, f'best-model-parameters_{folder_idx}.pt')

            train(model, train_loader, earling_stopping, optim, criterion, self.device, scheduler, best_model_path,
                  epochs=self.num_epochs, use_cuda=self.use_cuda)

            # load best model
            model = create_model()
            model.load_state_dict(torch.load(best_model_path))
            model = model.to(self.device)

            metrics = test(model, test_loader, self.device, use_cuda=self.use_cuda)

            for k, v in metrics.items():
                if k != 'confusion_matrix':
                    logging.info('Fold {} {}: {}\n'.format(folder_idx + 1, k.capitalize(), v))

            cum_acc.append(metrics['accuracy'])
            cum_f1.append(metrics['f1-score'])
            cum_recall.append(metrics['recall'])
            cum_precision.append(metrics['precision'])
            cum_auc.append(metrics['roc_auc'])

        self.print_metrics(cum_acc, cum_recall, cum_precision, cum_auc, cum_f1)


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


def load_data(filepath):
    # Load data
    dataset = np.load(filepath, allow_pickle=True)
    X = rescale(dataset["X"])
    y = dataset["y"]
    y_col_names = list(dataset['y_col_names'])
    col_idx_target = y_col_names.index(clin_variable_target)
    logging.info(f"Clinical variable: {clin_variable_target}")

    X, y = clean(X, y, col_idx_target)
    y_target = y[:, col_idx_target]
    return X, y, y_target


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
    logging.getLogger().addHandler(console)


def create_model():
    model = TimeSeriesTransformer(batch_first=False,
                                  dim_val=512,
                                  n_encoder_layers=6,
                                  n_heads=8,
                                  dropout_encoder=0.1,
                                  dim_feedforward_encoder=128)
    return model


def get_class_distribution(class_info):
    names, quant = class_info
    str = ""
    for name, q in zip(names, quant):
        str += f"{name}: {q/sum(quant)*100:.2f}% "
    return str


def augment_data(train_data, train_labels):
    train_augmented = []
    train_labels_augmented = []
    logging.info("Data augmentation generation")
    for sample, label in tqdm(zip(train_data, train_labels), total=len(train_data)):
        sample_rot = DA_Rotation(sample)
        train_augmented.append(sample_rot)
        train_labels_augmented.append(label)

        sample_per = DA_Permutation(sample, minSegLength=50)
        train_augmented.append(sample_per)
        train_labels_augmented.append(label)

        sample_time = DA_TimeWarp(sample)
        train_augmented.append(sample_time)
        train_labels_augmented.append(label)

        sample_mag = DA_MagWarp(sample, sigma=0.2)
        train_augmented.append(sample_mag)
        train_labels_augmented.append(label)

        sample_jit = DA_Jitter(sample)
        train_augmented.append(sample_jit)
        train_labels_augmented.append(label)

        sample_sca = DA_Scaling(sample)
        train_augmented.append(sample_sca)
        train_labels_augmented.append(label)

        sample_rand = DA_RandSampling(sample)
        train_augmented.append(sample_rand)
        train_labels_augmented.append(label)

    train_data = np.concatenate((train_data, np.array(train_augmented)))
    train_labels = np.concatenate((train_labels, np.array(train_labels_augmented)))
    return train_data, train_labels


if __name__ == '__main__':
    start = time()

    exp_name = f"exp_transformer_{time()}.log"
    out_folder = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/"
    best_model_folder = "/home/jsenadesouza/DA-healthy2patient/results/best_models/pain_transformer"
    filepath = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/f10_t1800_IntelligentICU_PAIN_ADAPT.npz"

    num_epochs = 1
    batch_size = 40
    data_aug = True
    loss = "CE"
    weighted_sampler = True
    clin_variable_target = "pain_score_class"

    model = Transformer(filepath, exp_name, out_folder, num_epochs, batch_size, data_aug=data_aug, loss=loss,
                 weighted_sampler=weighted_sampler)
    model.run()