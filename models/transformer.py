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
import optuna
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder
# from dask.distributed import Client, wait
# from dask_cuda import LocalCUDACluster
# from multiprocessing import Manager
# from joblib import parallel_backend
import joblib
# cluster = LocalCUDACluster()
# client = Client(cluster)
random.seed(42)

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
        self.device = torch.device('cuda:2') if self.use_cuda else torch.device('cpu')
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

        # print variable info
        self.logger.info(f"Data augmentation: {self.data_aug}")
        self.logger.info(f"Loss: {self.loss}")
        self.logger.info(f"Weighted sampler: {self.weighted_sampler}")

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

        # train_labels = np.array([self.labels2idx[label] for label in train_labels])
        # test_labels = np.array([self.labels2idx[label] for label in test_labels])
        # val_labels = np.array([self.labels2idx[label] for label in val_labels])

        self.logger.info(f"Folder {folder_idx + 1}")
        self.logger.info(f"Train data: {get_class_distribution(np.unique(np.argmax(train_labels, axis=1), return_counts=True))}")
        self.logger.info(f"Test data: {get_class_distribution(np.unique(np.argmax(test_labels, axis=1), return_counts=True))}")
        self.logger.info(f"Val data: {get_class_distribution(np.unique(np.argmax(val_labels, axis=1), return_counts=True))}")

        return train_data, train_labels, test_data, test_labels, val_data, val_labels

    def get_loaders(self, train_data, train_labels, test_data, test_labels, val_data, val_labels):
        train_set = SensorDataset(train_data, None, train_labels, dataaug=self.data_aug)
        test_set = SensorDataset(test_data, None, test_labels)
        val_set = SensorDataset(val_data, None, val_labels)

        if self.weighted_sampler:
            class_sample_count = np.array(
                [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in np.argmax(train_labels, axis=1)])

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

    def objective(self, trial):

        # Split data
        patients = list(np.unique(self.y[:, -1]))
        random.shuffle(patients)
        patient_splits = np.array_split(patients, 6)

        # split the data into train, val and test sets
        train_data, train_labels, test_data, test_labels, val_data, val_labels = self.split_data(patient_splits,
                                                                                                 0)
        # get dataloaders
        train_loader, test_loader, val_loader = self.get_loaders(train_data, train_labels, test_data, test_labels,
                                                                 val_data, val_labels)
        #dim_val = trial.suggest_categorical("dim_val", [512, 768, 1024])
        dim_val = 512
        n_encoder_layers = trial.suggest_categorical("n_encoder_layers", [4,5,6,7,8])
        n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])
        dim_feedforward_encoder = trial.suggest_categorical("dim_feedforward_encoder", [512, 1024, 2048])

        model = create_model(dim_val=dim_val, n_encoder_layers=n_encoder_layers, n_heads=n_heads, dim_feedforward_encoder=dim_feedforward_encoder)
        model = model.to(self.device)
        self.loss = trial.suggest_categorical('loss', ['focal', 'bce'])
        if self.loss == "focal":
            alpha = trial.suggest_float('alpha', 0.1, 0.9, step=0.1)
            gama = trial.suggest_int('gama', 1, 5)
            criterion = FocalLoss(alpha=alpha, gamma=gama)
        else:
            pos = trial.suggest_categorical('pos_weight', [True, False])
            #class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            #class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            if pos:
                pos_weight = (train_labels == 0.).sum() / train_labels.sum()
                pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(self.device)

                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
            else:
                criterion = nn.BCEWithLogitsLoss(reduction='mean')
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
        earling_stopping = {"val_loader": test_loader, "last_loss": 100, "patience": 10, "triggertimes": 0}
        best_model_folder = os.path.join(self.out_folder, "best_model")
        if not os.path.exists(best_model_folder):
            os.makedirs(best_model_folder)
        best_model_path = os.path.join(best_model_folder, f'best-model-parameters_{time()}.pt')

        train(model, train_loader, earling_stopping, optim, criterion, self.device, scheduler, best_model_path, trial,
              epochs=self.num_epochs, use_cuda=self.use_cuda)

        # load best model

        #model = create_model(dim_val=dim_val, n_encoder_layers=n_encoder_layers, n_heads=n_heads, dim_feedforward_encoder=dim_feedforward_encoder)
        #checkpoint = torch.load(best_model_path, map_location=lambda storage, loc: storage.cuda(self.device))
        #model.load_state_dict(checkpoint["model"])
        #model = model.to(self.device)

        metrics = test(model, test_loader, self.device, use_cuda=self.use_cuda)

        return metrics['f1-score_macro']

    def loss_weights(self, msk, device):
        weights = [int(sum(msk == 0)),
                   int(sum(msk == 1))]
        if weights[0] == 0 or weights[1] == 0:
            weights = torch.FloatTensor([1.0, 1.0]).to(device)
        else:
            if weights[0] > weights[1]:
                weights = torch.FloatTensor([1.0, weights[0] / weights[1]]).to(device)
            else:
                weights = torch.FloatTensor([weights[1] / weights[0], 1.0]).to(device)

        return weights

    def run(self, num_folders):

        cum_acc, cum_recall, cum_precision, cum_auc, cum_f1 = [], [], [], [], []
        cum_recall_macro, cum_precision_macro, cum_f1_macro = [], [], []
        self.print_info()

        # Split data
        patients = list(np.unique(self.y[:, -1]))
        random.shuffle(patients)
        patient_splits = np.array_split(patients, 6)

        for folder_idx in range(num_folders):

            writer = SummaryWriter(
                f'/home/jsenadesouza/DA-healthy2patient/results/outcomes/tensorboard/transformers/self.exp_name/run_{time()}')

            # split the data into train, val and test sets
            train_data, train_labels, test_data, test_labels, val_data, val_labels = self.split_data(patient_splits,
                                                                                                     folder_idx)
            # get dataloaders
            train_loader, test_loader, val_loader = self.get_loaders(train_data, train_labels, test_data, test_labels,
                                                                     val_data, val_labels)

            model = create_model(self.n_classes)
            model = model.to(self.device)
            if self.loss == "focal":
                criterion = FocalLoss(alpha=0.5, gamma=2)
            else:
                # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(train_labels, axis=1)), y=np.argmax(train_labels, axis=1))
                # class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
                # class_weights_hugo = self.loss_weights(np.argmax(train_labels, axis=1), self.device)
                class_weights = torch.tensor([1., 1000.], dtype=torch.float).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-04)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
            earling_stopping = {"val_loader": test_loader, "last_loss": 100, "patience": 10, "triggertimes": 0}
            best_model_folder = os.path.join(self.out_folder, "best_model")
            if not os.path.exists(best_model_folder):
                os.makedirs(best_model_folder)
            best_model_path = os.path.join(best_model_folder, f'best-model-parameters_{folder_idx}.pt')

            train(model, train_loader, earling_stopping, optim, criterion, self.device, scheduler, best_model_path,
                  writer, trial=None, epochs=self.num_epochs, use_cuda=self.use_cuda)

            # load best model

            model = create_model(self.n_classes)
            checkpoint = torch.load(best_model_path, map_location=lambda storage, loc: storage.cuda(self.device))
            model.load_state_dict(checkpoint["model"])
            model = model.to(self.device)

            metrics = test(model, test_loader, self.device, use_cuda=self.use_cuda)

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

    def get_clinical_data(self, y, y_col_names, target_col_name):
        regression_val = [0, 2, 6, 8, 10, 14, 16]
        col_target = y_col_names.index(target_col_name)
        col_target_reg = y_col_names.index(target_col_name.split("_class")[0])

        clin_var_idx = []
        for ii in regression_val:
            ii = int(ii)
            if ii != col_target and ii != col_target_reg:
                clin_var_idx.append(ii)

        clin_var = y[:, clin_var_idx]

        self.logger.info(f'Target = {y_col_names[col_target]}\n')
        self.logger.info("\nCLinical variables used:")
        for idx in clin_var_idx:
            self.logger.info(f"{y_col_names[idx]}")

        return clin_var.astype(np.float32)

    def load_data(self, filepath):
        # Load data
        dataset = np.load(filepath, allow_pickle=True)
        X = dataset["X"]
        y = dataset["y"]
        y_col_names = list(dataset['y_col_names'])
        col_idx_target = y_col_names.index(clin_variable_target)

        X, y = clean(X, y, col_idx_target)
        y_target = y[:, col_idx_target]
        enc = OneHotEncoder(handle_unknown='ignore')
        y_target = enc.fit_transform(y_target.reshape(-1, 1)).toarray()
        return X, y, y_target


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


def create_model(n_classes, dim_val=512, n_encoder_layers=6, n_heads=8, dropout_encoder=0.1, dim_feedforward_encoder=128):
    model = TimeSeriesTransformer(n_classes,
                                  batch_first=False,
                                  dim_val=dim_val,
                                  n_encoder_layers=n_encoder_layers,
                                  n_heads=n_heads,
                                  dropout_encoder=dropout_encoder,
                                  dim_feedforward_encoder=dim_feedforward_encoder)
    return model


def get_class_distribution(class_info):
    names, quant = class_info
    str = ""
    for name, q in zip(names, quant):
        str += f"{name}: {q/sum(quant)*100:.2f}% "
    return str

def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )


def optuna_search():
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("/home/jsenadesouza/DA-healthy2patient/results/study1.log", mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    sampler = optuna.samplers.TPESampler(seed=42)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction="maximize", sampler=sampler)
    logger.info("Start optimization.")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(model.objective, n_trials=50, show_progress_bar=True,
                   gc_after_trial=True)  # callbacks=[logging_callback])# gc_after_trial=Trueif running out of memory
    # with Manager() as manager:
    #     # Initialize the queue by adding available GPU IDs.
    #     n_gpu = 2
    #     gpu_queue = manager.Queue()
    #     for i in range(n_gpu):
    #         gpu_queue.put(i)
    #     with parallel_backend("dask", n_jobs=n_gpu):
    #         study.optimize(model.objective, n_trials=50, callbacks=[logging_callback], n_jobs=n_gpu)

    joblib.dump(study, "/home/jsenadesouza/DA-healthy2patient/results/study.pkl")
    # Create a dataframe from the study.
    df = study.trials_dataframe()
    df.to_csv("/home/jsenadesouza/DA-healthy2patient/results/study.csv")

    print("Number of finished trials: ", len(study.trials))

if __name__ == '__main__':
    start = time()

    exp_name = f"exp_transformer_{time()}.log"
    out_folder = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/"
    best_model_folder = "/home/jsenadesouza/DA-healthy2patient/results/best_models/pain_transformer_15min"
    filepath = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/f10_t900_IntelligentICU_PAIN_ADAPT_15min.npz"

    num_epochs = 200
    batch_size = 40
    data_aug = False
    loss = "ce"
    weighted_sampler = False
    clin_variable_target = "pain_score_class"
    num_folders = 1

    model = Transformer(filepath, exp_name, out_folder, num_epochs, batch_size, data_aug=data_aug, loss=loss,
                 weighted_sampler=weighted_sampler)

    model.run(num_folders)


