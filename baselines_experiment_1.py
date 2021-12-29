import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as st
from torch.utils.data import DataLoader

from utils.data import SensorDataset
from utils.models import MetaSenseModel, MiniMetaSenseModel
from utils.utils import train, test


def run_baseline0(args):
    dataset = np.load(args.filepath, allow_pickle=True)
    labels2idx = {k: idx for idx, k in enumerate(np.unique(dataset['y_test']))}

    folds = dataset['kfold_{0}_shot'.format(args.num_shots)]

    cum_acc, cum_f1, cum_recall, cum_conf_matrices = [], [], [], []

    for fold_idx, fold in enumerate(folds):
        train_data, train_labels = dataset['X_test'][fold['train_idx']].squeeze(), dataset['y_test'][fold['train_idx']]
        test_data, test_labels = dataset['X_test'][fold['test_idx']].squeeze(), dataset['y_test'][fold['test_idx']]

        train_labels = np.array([labels2idx[label] for label in train_labels])
        test_labels = np.array([labels2idx[label] for label in test_labels])

        train_set = SensorDataset(train_data, train_labels)
        test_set = SensorDataset(test_data, test_labels)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        model = MetaSenseModel(args.num_ways)

        if args.use_cuda:
            model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        train(model, train_loader, optim, criterion, epochs=args.num_epochs, use_cuda=args.use_cuda)

        metrics = test(model, test_loader, use_cuda=args.use_cuda)

        print()
        for k, v in metrics.items():
            if k == 'confusion_matrix':
                print('Fold {} {}: {}'.format(fold_idx + 1, k.capitalize(), v))
            else:
                print('Fold {} {}: {:.04f}'.format(fold_idx + 1, k.capitalize(), v))

        cum_acc.append(metrics['accuracy'])
        cum_f1.append(metrics['f1-score'])
        cum_recall.append(metrics['recall'])
        cum_conf_matrices.append(metrics['confusion_matrix'])

    ci_mean = st.t.interval(0.9, len(cum_acc) - 1, loc=np.mean(cum_acc), scale=st.sem(cum_acc))
    ci_f1 = st.t.interval(0.9, len(cum_f1) -1, loc=np.mean(cum_f1), scale=st.sem(cum_f1))
    ci_recall = st.t.interval(0.9, len(cum_recall) -1, loc=np.mean(cum_recall), scale=st.sem(cum_recall))
    confusion_matrix = sum(cum_conf_matrices)

    return {
        'accuracy': '{:.2f} ± {:.2f}'.format(np.mean(cum_acc) * 100, abs(np.mean(cum_acc) - ci_mean[0]) * 100),
        'f1-score': '{:.2f} ± {:.2f}'.format(np.mean(cum_f1) * 100, abs(np.mean(cum_f1) - ci_f1[0]) * 100),
        'recall': '{:.2f} ± {:.2f}'.format(np.mean(cum_recall) * 100, abs(np.mean(cum_recall) - ci_recall[0]) * 100),
        #'confusion matrix': str(confusion_matrix)
    }

def run_baseline1(args):
    dataset = np.load(args.filepath, allow_pickle=True)
    folds = dataset['kfold_no_shot']
    base1_results = {}
    for data_source in dataset['Xy_train'][0].keys():

        train_data = np.array(dataset['Xy_train'][0][data_source][0])
        _train_labels = np.array(dataset['Xy_train'][0][data_source][1])

        source_labels2idx = {k: idx for idx, k in enumerate(np.unique(_train_labels))}

        cum_acc, cum_f1, cum_recall, cum_conf_matrices = [], [], [], []

        for fold_idx, fold in enumerate(folds):
            test_data, test_labels = dataset['X_test'][fold['test_idx']].squeeze(), dataset['y_test'][fold['test_idx']]
            target_labels2idx = {k: idx for idx, k in enumerate(np.unique(test_labels))}

            train_labels = np.array([source_labels2idx[label] for label in _train_labels])
            test_labels = np.array([target_labels2idx[label] for label in test_labels])

            train_set = SensorDataset(train_data, train_labels)
            test_set = SensorDataset(test_data, test_labels)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

            model = MetaSenseModel(args.num_ways)

            if args.use_cuda:
                model = model.cuda()

            criterion = nn.CrossEntropyLoss()
            optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            train(model, train_loader, optim, criterion, epochs=args.num_epochs, use_cuda=args.use_cuda)

            metrics = test(model, test_loader, use_cuda=args.use_cuda)

            for k, v in metrics.items():
                if k == 'confusion_matrix':
                    print('Fold {} {}: {}'.format(fold_idx + 1, k.capitalize(), v))
                else:
                    print('Fold {} {}: {:.04f}'.format(fold_idx + 1, k.capitalize(), v))

            cum_acc.append(metrics['accuracy'])
            cum_f1.append(metrics['f1-score'])
            cum_recall.append(metrics['recall'])
            #cum_conf_matrices.append(metrics['confusion_matrix'])

        ci_mean = st.t.interval(0.9, len(cum_acc) - 1, loc=np.mean(cum_acc), scale=st.sem(cum_acc))
        ci_f1 = st.t.interval(0.9, len(cum_f1) -1, loc=np.mean(cum_f1), scale=st.sem(cum_f1))
        ci_recall = st.t.interval(0.9, len(cum_recall) -1, loc=np.mean(cum_recall), scale=st.sem(cum_recall))
        #confusion_matrix = sum(cum_conf_matrices)

        base1_results[f'baseline1_no_shot_{data_source}'] = {
            'accuracy': '{:.2f} ± {:.2f}'.format(np.mean(cum_acc) * 100, abs(np.mean(cum_acc) - ci_mean[0]) * 100),
            'f1-score': '{:.2f} ± {:.2f}'.format(np.mean(cum_f1) * 100, abs(np.mean(cum_f1) - ci_f1[0]) * 100),
            'recall': '{:.2f} ± {:.2f}'.format(np.mean(cum_recall) * 100, abs(np.mean(cum_recall) - ci_recall[0]) * 100),
            #'confusion matrix': str(confusion_matrix)
        }

    return base1_results

def freeze(model,bn_freeze=True):
    for name,param in model.named_parameters():
        if bn_freeze:
            param.requires_grad = False
        elif name.find('bn') == -1:
            param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def run_baseline2(args):
    device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    dataset = np.load(args.filepath, allow_pickle=True)

    source_datasets = list(dataset['Xy_train'][0].keys())
    folds = dataset['kfold_{0}_shot'.format(args.num_shots)]

    results = {}

    for source in source_datasets:
        print(f'Training using {source} as source dataset...\n', flush=True)
        # Train using as source each of the datasets.
        train_data, train_labels = dataset['Xy_train'][0][source]
        train_data, train_labels = np.array(train_data).squeeze(), np.array(train_labels)
        source_labels2idx = {k: idx for idx, k in enumerate(np.unique(train_labels))}

        train_labels = np.array([source_labels2idx[label] for label in train_labels])
        train_set = SensorDataset(train_data, train_labels)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

        model = MetaSenseModel(len(source_labels2idx))

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        if args.use_cuda:
            model = model.cuda()

        train(model, train_loader, optim, criterion, epochs=args.num_epochs, use_cuda=args.use_cuda)
        outpath = os.path.join(args.output, args.filepath.split(os.sep)[-1][:-4])
        checkpoint_filepath = outpath + '.pth'
        with open(checkpoint_filepath, 'wb') as f:
            torch.save(model.state_dict(), f)

        # Adapt for each fold of the cross-validation
        cum_acc, cum_f1, cum_recall = [], [], []

        for fold_idx, fold in enumerate(folds):
            print(f'Finetuning to target....\n', flush=True)
            model = MetaSenseModel(len(source_labels2idx))

            labels2idx = {k: idx for idx, k in enumerate(np.unique(dataset['y_test']))}
            
            # Reload model weights
            with open(checkpoint_filepath, 'rb') as f:
                model.load_state_dict(torch.load(f, map_location=device))

            model.classifier[-1] = nn.Linear(64, len(labels2idx))
            freeze(model.features)

            if args.use_cuda:
                model = model.cuda()
            
            train_data, train_labels = dataset['X_test'][fold['train_idx']].squeeze(), dataset['y_test'][fold['train_idx']]
            test_data, test_labels = dataset['X_test'][fold['test_idx']].squeeze(), dataset['y_test'][fold['test_idx']]

            train_labels = np.array([labels2idx[label] for label in train_labels])
            test_labels = np.array([labels2idx[label] for label in test_labels])

            train_set = SensorDataset(train_data, train_labels)
            test_set = SensorDataset(test_data, test_labels)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

            ft_optim = torch.optim.Adam(model.parameters(), lr=5e-4)
            #ft_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            train(model, train_loader, ft_optim, criterion, epochs=25, use_cuda=args.use_cuda)

            metrics = test(model, test_loader, use_cuda=args.use_cuda)

            for k, v in metrics.items():
                if k == 'confusion_matrix':
                    print('Fold {} {}: {}'.format(fold_idx + 1, k.capitalize(), v))
                else:
                    print('Fold {} {}: {:.04f}'.format(fold_idx + 1, k.capitalize(), v))

            cum_acc.append(metrics['accuracy'])
            cum_f1.append(metrics['f1-score'])
            cum_recall.append(metrics['recall'])
            #cum_conf_matrices.append(metrics['confusion_matrix'])

        ci_mean = st.t.interval(0.9, len(cum_acc) - 1, loc=np.mean(cum_acc), scale=st.sem(cum_acc))
        ci_f1 = st.t.interval(0.9, len(cum_f1) -1, loc=np.mean(cum_f1), scale=st.sem(cum_f1))
        ci_recall = st.t.interval(0.9, len(cum_recall) -1, loc=np.mean(cum_recall), scale=st.sem(cum_recall))
        #confusion_matrix = sum(cum_conf_matrices)

        results[f'baseline2_{source}_{args.num_shots}_shot'] = {
            'accuracy': '{:.2f} ± {:.2f}'.format(np.mean(cum_acc) * 100, abs(np.mean(cum_acc) - ci_mean[0]) * 100),
            'f1-score': '{:.2f} ± {:.2f}'.format(np.mean(cum_f1) * 100, abs(np.mean(cum_f1) - ci_f1[0]) * 100),
            'recall': '{:.2f} ± {:.2f}'.format(np.mean(cum_recall) * 100, abs(np.mean(cum_recall) - ci_recall[0]) * 100)
            #'confusion matrix': str(confusion_matrix)
        }

    return results  


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model baselines')

    # General
    parser.add_argument('filepath', type=str, help='Path to `.npz` file.')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output path folder')

    # Optimization
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Number of instances per batch')
    parser.add_argument('-l', '--learning-rate', type=int, default=1e-3, help='Epoch learning rate')
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help='Number of training epochs')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers to use for data-loading')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--base0', action='store_true')
    parser.add_argument('--base1', action='store_true')
    parser.add_argument('--base2', action='store_true')

    args = parser.parse_args()
    args.num_ways = 3

    args.use_cuda = torch.cuda.is_available()
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    else:
        args.output = './'

    results = {}

    # run baseline 0
    if args.base0:
        print('Running baseline 0\n', flush=True)
        for num_shots in ['no', 1, 5, 10]:
            args.num_shots = num_shots
            results[f'baseline0_{num_shots}_shot'] = run_baseline0(args)

    # run baseline 1
    if args.base1:
        print("Running baseline 1\n", flush=True)
        # cross dataset WITHOUT transfer learning
        results.update(run_baseline1(args))

    # run baseline 2
    if args.base2:
        print("Running baseline 2\n", flush=True)
        # cross dataset WITH transfer learning
        for num_shots in ['no', 1, 5, 10]:
        #for num_shots in [1]:
            print("Shot: {}\n".format(num_shots), flush=True)
            args.num_shots = num_shots
            results.update(run_baseline2(args))

    table = {'experiment': []}
    for baseline, result in results.items():
        table['experiment'].append(baseline)
        for k, v in result.items():
            if k not in table:
                table[k] = []
            table[k].append(v)

    results_df = pd.DataFrame(table)
    outpath = os.path.join(args.output, args.filepath.split(os.sep)[-1][:-4]) + '.txt'
    with open(outpath, 'w') as f:
        f.write(results_df.to_markdown())
        print(results_df.to_markdown())
