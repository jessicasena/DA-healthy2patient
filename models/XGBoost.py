import sys
import os
from multiprocessing.dummy import Pool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as st
from time import time
import logging
import random
from utils.utils import get_metrics
from xgboost import XGBClassifier
from collections import Counter
import math

random.seed(42)

def magnitude(sample):
    mag_vector = []
    for s in sample:
        mag_vector.append(math.sqrt(sum([s[0]**2, s[1]**2, s[2]**2])))
    return mag_vector

def A(sample):
    feat = []
    for col in range(0,sample.shape[1]):
        average = np.average(sample[:, col])
        feat.append(average)

    return np.mean(feat)


def SD(sample):
    feat = []
    for col in range(0, sample.shape[1]):
        std = np.std(sample[:, col])
        feat.append(std)

    return np.mean(feat)


def AAD(sample):
    feat = []
    for col in range(0, sample.shape[1]):
        data = sample[col,:]
        add = np.mean(np.absolute(data - np.mean(data)))
        feat.append(add)

    return np.mean(feat)


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
    return np.mean(feat)

def COR(sample):
    feat = []
    for axis_i in range(0, sample.shape[1]):
        for axis_j in range(axis_i+1, sample.shape[1]):
            cor = np.corrcoef(sample[:, axis_i], sample[:, axis_j])
            cor = 0 if np.isnan(cor) else cor[0][1]
            feat.append(cor)

    return np.mean(feat)


def mag_mean(sample):
    mag = magnitude(sample)
    ft_mean = np.mean(mag)
    return ft_mean

def mag_std(sample):
    mag = magnitude(sample)
    ft_std = np.std(mag)
    return ft_std


def feature_extraction(sample):
    """
    Derive three activity intensity cues: mean and standard deviation of activity intensity,
    and duration of immobility during assessment window to summarize the data.
    # Average - A,
    # Standard Deviation - SD,
    # Average Absolute Difference - AAD,
    # Average Resultant Acceleration - ARA(1),
    """
    features = []

    mag = magnitude(sample)
    features = np.mean(mag)
    features = np.hstack((features, np.std(mag)))
    features = np.hstack((features, A(sample)))
    features = np.hstack((features, SD(sample)))
    features = np.hstack((features, AAD(sample)))
    features = np.hstack((features, ARA(sample)))

    return features

class XGboost:
    def __init__(self, data_path, exp_name, out_folder):

        self.data_path = data_path
        self.exp_name = exp_name
        self.out_folder = out_folder

        self.logger = set_logger(os.path.join(self.out_folder, self.exp_name))
        self.X, self.y, self.y_target = self.load_data(self.data_path)
        self.n_classes = np.unique(self.y_target).shape[0]
        self.labels2idx = {k: idx for idx, k in enumerate(np.unique(self.y_target))}

    def print_info(self):
        self.logger.info(f"Experiment name: {self.exp_name}")
        self.logger.info(f'Total of samples: {self.y.shape[0]}\n')
        self.logger.info(f'Total of hours: {self.y.shape[0] * 30 / 60}\n')
        pain_samples = np.count_nonzero(np.char.find(np.array(self.y[:, -1]), "P") != -1)
        adapt_samples = np.count_nonzero(np.char.find(np.array(self.y[:, -1]), "I") != -1)
        self.logger.info(
            f"Intelligent ICU samples: {len(self.y) - pain_samples - adapt_samples}, PAIN samples: {pain_samples}, "
            f"ADAPT samples: {adapt_samples}")
        self.logger.info(f'Total of patients: {len(np.unique(self.y[:, -1]))}\n')
        pain_patients = np.count_nonzero(np.char.find(np.unique(self.y[:, -1]), "P") != -1)
        adapt_patients = np.count_nonzero(np.char.find(np.unique(self.y[:, -1]), "I") != -1)
        self.logger.info(
            f'Intelligent ICU patients: {len(np.unique(self.y[:, -1])) - adapt_patients - pain_patients}, '
            f'PAIN patients: {pain_patients}, ADAPT patients: {adapt_patients}\n')
        self.logger.info(str(np.unique(self.y_target, return_counts=True)) + "")


    def print_metrics(self, cum_acc, cum_recall, cum_precision, cum_auc, cum_f1, cum_recall_macro, cum_precision_macro, cum_f1_macro):
        current_acc = np.array(cum_acc)
        current_auc = np.array(cum_auc)
        current_recall_macro = np.array(cum_recall_macro)
        current_prec_macro = np.array(cum_precision_macro)
        current_f1_macro = np.array(cum_f1_macro)

        ci_mean = st.t.interval(0.95, len(current_acc) - 1, loc=np.mean(current_acc), scale=st.sem(current_acc))
        ci_auc = st.t.interval(0.95, len(current_auc) - 1, loc=np.mean(current_auc), scale=st.sem(current_auc))
        ci_recall_macro = st.t.interval(0.95, len(current_recall_macro) - 1, loc=np.mean(current_recall_macro),
                                        scale=st.sem(current_recall_macro))
        ci_prec_macro = st.t.interval(0.95, len(current_prec_macro) - 1, loc=np.mean(current_prec_macro),
                                      scale=st.sem(current_prec_macro))
        ci_f1_macro = st.t.interval(0.95, len(current_f1_macro) - 1, loc=np.mean(current_f1_macro),
                                    scale=st.sem(current_f1_macro))

        self.logger.info('accuracy: {:.2f} ± {:.2f}\n'.format(np.mean(current_acc) * 100,
                                                              abs(np.mean(current_acc) - ci_mean[0]) * 100))

        self.logger.info('recall_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall_macro) * 100,
                                                                  abs(np.mean(current_recall_macro) - ci_recall_macro[
                                                                      0]) * 100))
        self.logger.info('precision_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec_macro) * 100,
                                                                     abs(np.mean(current_prec_macro) - ci_prec_macro[
                                                                         0]) * 100))
        self.logger.info('f1-score_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1_macro) * 100,
                                                                    abs(np.mean(current_f1_macro) - ci_f1_macro[
                                                                        0]) * 100))
        self.logger.info('roc_auc: {:.2f} ± {:.2f}\n'.format(np.mean(current_auc) * 100,
                                                             abs(np.mean(current_auc) - ci_auc[0]) * 100))

        for class_ in range(len(np.unique(self.y_target))):
            self.logger.info(f"Class: {class_}")

            current_f1 = np.array(cum_f1)[:, class_]
            current_recall = np.array(cum_recall)[:, class_]
            current_prec = np.array(cum_precision)[:, class_]

            ci_f1 = st.t.interval(0.95, len(current_f1) - 1, loc=np.mean(current_f1), scale=st.sem(current_f1))
            ci_recall = st.t.interval(0.95, len(current_recall) - 1, loc=np.mean(current_recall),
                                      scale=st.sem(current_recall))
            ci_prec = st.t.interval(0.95, len(current_prec) - 1, loc=np.mean(current_prec), scale=st.sem(current_prec))

            self.logger.info('recall: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall) * 100,
                                                            abs(np.mean(current_recall) - ci_recall[0]) * 100))
            self.logger.info('precision: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec) * 100,
                                                               abs(np.mean(current_prec) - ci_prec[0]) * 100))
            self.logger.info('f1-score: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1) * 100,
                                                              abs(np.mean(current_f1) - ci_f1[0]) * 100))



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

        self.logger.info(f"Folder {folder_idx + 1}")
        self.logger.info(f"Train data: {get_class_distribution(np.unique(train_labels, return_counts=True))}")
        self.logger.info(f"Test data: {get_class_distribution(np.unique(test_labels, return_counts=True))}")
        self.logger.info(f"Val data: {get_class_distribution(np.unique(val_labels, return_counts=True))}")

        return train_data, train_labels, test_data, test_labels, val_data, val_labels

    def run(self, num_folders):

        cum_acc, cum_recall, cum_precision, cum_auc, cum_f1 = [], [], [], [], []
        cum_recall_macro, cum_precision_macro, cum_f1_macro = [], [], []
        self.print_info()

        # Split data
        patients = list(np.unique(self.y[:, -1]))
        random.shuffle(patients)
        patient_splits = np.array_split(patients, 6)

        for folder_idx in range(num_folders):
            # writer = SummaryWriter(
            #     f'/home/jsenadesouza/DA-healthy2patient/results/outcomes/tensorboard/transformers/self.exp_name/run_{time()}')

            # split the data into train, val and test sets
            train_data, train_labels, test_data, test_labels, val_data, val_labels = self.split_data(patient_splits,
                                                                                                     folder_idx)
            counter = Counter(train_labels)
            # estimate scale_pos_weight value
            estimate = counter[0] / counter[1]
            print(f'Estimate: {estimate}. {counter}')
            # fit model no training data
            model = XGBClassifier(tree_method="gpu_hist", use_label_encoder=False, eval_metric='logloss',
                                  scale_pos_weight=estimate)
            model.fit(train_data, train_labels)

            y_pred = model.predict(test_data)
            y_pred = [round(value) for value in y_pred]
            metrics = get_metrics(test_data, y_pred)

            for k, v in metrics.items():
                if k != 'confusion_matrix':
                    self.logger.info('Fold {} {}: {}\n'.format(folder_idx + 1, k.capitalize(), v))

            cum_acc.append(metrics['accuracy'])
            cum_f1.append(metrics['f1-score'])
            cum_recall.append(metrics['recall'])
            cum_precision.append(metrics['precision'])
            cum_auc.append(metrics['roc_auc'])
            cum_f1_macro.append(metrics['f1-score_macro'])
            cum_recall_macro.append(metrics['recall_macro'])
            cum_precision_macro.append(metrics['precision_macro'])

        self.print_metrics(cum_acc, cum_recall, cum_precision, cum_auc, cum_f1, cum_recall_macro, cum_precision_macro, cum_f1_macro)


    def load_data(self, filepath):
        # Load data
        dataset = np.load(filepath, allow_pickle=True)
        X = rescale(dataset["X"])
        y = dataset["y"]
        y_col_names = list(dataset['y_col_names'])
        col_idx_target = y_col_names.index(clin_variable_target)
        #self.logger.info(f"Clinical variable: {clin_variable_target}")

        X, y = clean(X, y, col_idx_target)
        y_target = y[:, col_idx_target]

        X_trasp = np.transpose(np.squeeze(X), (0, 1, 2))
        print("Extracting Features")
        start = time()
        with Pool(30) as p:
            X_feat = p.map(feature_extraction, X_trasp)
        end = time()
        print(f"{end - start:.4} seconds passed.")

        X_feat = np.array(X_feat)
        return X_feat, y, y_target

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

if __name__ == '__main__':
    start = time()

    exp_name = f"exp_transformer_{time()}.log"
    out_folder = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/"
    best_model_folder = "/home/jsenadesouza/DA-healthy2patient/results/best_models/pain_transformer_15min"
    filepath = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/f10_t900_IntelligentICU_PAIN_ADAPT_15min.npz"

    num_epochs = 100
    batch_size = 40
    data_aug = False
    loss = "focal"
    weighted_sampler = True
    clin_variable_target = "pain_score_class"
    num_folders = 5

    model = XGboost(filepath, exp_name, out_folder)

    model.run(num_folders)


