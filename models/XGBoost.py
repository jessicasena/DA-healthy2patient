import sys
import os
from multiprocessing.dummy import Pool

sys.path.append("/home/jsenadesouza/DA-healthy2patient/code/")

import numpy as np

from time import time
import logging
import random
from utils.util import get_metrics, print_metrics
from models.util import load_data, set_logger, split_data, Find_Optimal_Cutoff
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import confusion_matrix
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
    mag = magnitude(sample)
    features = np.mean(mag)
    features = np.hstack((features, np.std(mag)))
    features = np.hstack((features, A(sample)))
    features = np.hstack((features, SD(sample)))
    features = np.hstack((features, AAD(sample)))
    features = np.hstack((features, ARA(sample)))

    return features

def plot_accel(sample, title, path):
    import matplotlib.pyplot as plt

    plt.plot(sample[:, 0])
    plt.plot(sample[:, 1])
    plt.plot(sample[:, 2])
    plt.title(os.path.join(path,title))

    plt.savefig(f"{title}.png")


class XGboost:
    def __init__(self, data_path, exp_name, out_folder):

        self.data_path = data_path
        self.exp_name = exp_name
        self.out_folder = out_folder
        self.clin_variable_target = "pain_score_class"

        self.logger = set_logger(os.path.join(self.out_folder, self.exp_name))
        self.X, self.y, self.y_target = self.load_data(self.data_path)
        self.n_classes = np.unique(self.y_target).shape[0]
        self.labels2idx = {k: idx for idx, k in enumerate(np.unique(self.y_target))}

    def load_data(self, data_path):
        X, y, y_target = load_data(data_path, clin_variable_target=self.clin_variable_target)

        X_trasp = np.transpose(np.squeeze(X), (0, 1, 2))
        print("Extracting Features")
        start = time()
        with Pool(100) as p:
            X_feat = p.map(feature_extraction, X_trasp)
        end = time()
        print(f"{end - start:.4} seconds passed.")

        X_feat = np.array(X_feat)
        return X_feat, y, y_target


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
            train_data, train_labels, test_data, test_labels, val_data, val_labels = split_data(self.X, self.y, self.y_target, self.labels2idx, patient_splits, folder_idx, logger=self.logger)
            counter = Counter(train_labels)
            # estimate scale_pos_weight value
            estimate = counter[0] / counter[1]
            print(f'Estimate: {estimate}. {counter}')
            # fit model no training data
            model = XGBClassifier(tree_method="gpu_hist", use_label_encoder=False, eval_metric='logloss',
                                  scale_pos_weight=estimate, random_state=42)
            model.fit(train_data, train_labels)

            y_pred = model.predict(test_data)
            y_score = model.predict_proba(test_data)

            threshold = Find_Optimal_Cutoff(test_labels, y_score[:, 1])
            print(f'threshold: {threshold}')
            y_pred_2 = list(map(lambda x: 1 if x > threshold else 0, y_score[:, 1]))
            print(confusion_matrix(test_labels, y_pred_2))

            def plot_metrics(ground_truth, predicted):
                metrics = get_metrics(ground_truth, predicted)
                for k, v in metrics.items():
                    if "confusion" in k:
                        logging.info('Fold {} {}:\n{}\n'.format(folder_idx, k.capitalize(), v))
                    else:
                        logging.info('Fold {} {}: {}\n'.format(folder_idx, k.capitalize(), v))
                return metrics

            metrics = plot_metrics(test_labels, y_pred)
            metrics = plot_metrics(test_labels, y_pred_2)

            cum_acc.append(metrics['accuracy'])
            cum_f1.append(metrics['f1-score'])
            cum_recall.append(metrics['recall'])
            cum_precision.append(metrics['precision'])
            cum_auc.append(metrics['roc_auc'])
            cum_f1_macro.append(metrics['f1-score_macro'])
            cum_recall_macro.append(metrics['recall_macro'])
            cum_precision_macro.append(metrics['precision_macro'])


        print_metrics(self.logger, self.n_classes, cum_acc, cum_recall, cum_precision, cum_auc, cum_f1, cum_recall_macro, cum_precision_macro, cum_f1_macro)


if __name__ == '__main__':
    start = time()

    exp_name = f"exp_transformer_{time()}.log"
    out_folder = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/"
    best_model_folder = "/home/jsenadesouza/DA-healthy2patient/results/best_models/pain_transformer_15min"
    filepath = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/t900_INTELLIGENT_PAIN_ADAPT_15min.npz"
    num_folders = 5
    model = XGboost(filepath, exp_name, out_folder)

    model.run(num_folders)


