import argparse
import datetime
import os
import random
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import learn2learn as l2l
import numpy as np
import scipy.stats as st
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.data import TargetDataset
from utils.models import MetaSenseModel
from utils.utils import (meta_test, multi_meta_train, multi_metaloaders,
                   save_plot_adapt_steps, balanced_dataset)


def main(args):
    start = time.time()
    num_workers = 0
    prefetch_factor = 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder_name = os.path.split(args.file_path)[-1].split('.npz')[0]
    results_path = os.path.join(args.results_path, folder_name)

    os.makedirs(results_path, exist_ok=True)

    dataset = np.load(args.file_path, allow_pickle=True)
    source_datasets = dataset['Xy_train'][0]

    # source_datasets = {}
    # # balance datasets
    # for name, (_data, _labels) in unbalanced_source_datasets.items():
    #     print(f"Source: {name}", flush=True)
    #     data, labels = balanced_dataset(_data, _labels)
    #     source_datasets[name] = [data, labels]

    metaloaders = multi_metaloaders(source_datasets, num_workers, prefetch_factor, args)

    # Create model
    model = MetaSenseModel(args.ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False)

    parameters = list(maml.parameters())
    opt = optim.Adam(parameters, args.meta_lr)

    # Setting scheduler.
    scheduler = torch.optim.lr_scheduler.StepLR(opt, args.iterations // 5, gamma=0.5)

    loss = nn.CrossEntropyLoss(reduction='mean')

    multi_meta_train(maml, device, loss, results_path, metaloaders, opt, scheduler, parameters, args)

    adapt_results = {}
    for i in range(5):
        args.fold = i
        tune_dataset = TargetDataset('tuning', dataset, args.fold, args.shots)
        tune_loader = DataLoader(tune_dataset, batch_size=args.train_bsz, num_workers=num_workers, shuffle=True,
                                 pin_memory=True, prefetch_factor=prefetch_factor, drop_last=False)

        test_dataset = TargetDataset('test', dataset, args.fold, args.shots)
        test_loader = DataLoader(test_dataset, batch_size=args.test_bsz, num_workers=num_workers, shuffle=False,
                                 pin_memory=True, prefetch_factor=prefetch_factor, drop_last=False)

        adapt_values = meta_test(maml, results_path, tune_loader, test_loader, loss, args.test_steps, args.tune_lr, device)

        for key, value in adapt_values.items():
            if key in adapt_results:
                adapt_results[key].append(value)

            else:
                adapt_results[key] = [value]

    end = time.time()
    duration = datetime.timedelta(seconds=end - start)

    # Save results
    outfile = open(os.path.join(results_path, f"results_5fold_{args.shots}shot.txt"), "w", buffering=1)
    outfile.write('Time: {}\n'.format(str(duration)))

    for met in ['accuracy', 'f1-score', 'recall']:

        adapt_values = []
        adapt_metric = []
        best_mean = -float("inf")
        best_metric = None
        best_losses = None
        best_adapt_steps = None

        for key, values in adapt_results.items():
            _metric = np.array(values)[:, 0]
            _metric = [x[met] for x in _metric]
            _losses = np.array(values)[:, 1]
            mean_acc = np.mean(_metric)

            adapt_values.append(key)
            adapt_metric.append(mean_acc)

            if mean_acc > best_mean:
                best_mean = mean_acc
                best_metric = _metric
                best_losses = _losses
                best_adapt_steps = key

        #confusion matrix
        _metric = np.array(adapt_results[best_adapt_steps])[:, 0]
        # sum_confusion = np.zeros((4, 4))
        # for x in _metric:
        #     sum_confusion = np.add(sum_confusion, x['confusion_matrix'])
        save_plot_adapt_steps(adapt_values, adapt_metric, results_path)

        ic_acc = st.t.interval(0.9, len(best_metric) - 1, loc=np.mean(best_metric), scale=st.sem(best_metric))
        ic_loss = st.t.interval(0.9, len(best_losses) - 1, loc=np.mean(best_losses), scale=st.sem(best_losses))

        outfile.write(f"\n{met}\n\n")
        outfile.write(f"Best value adapt steps: {best_adapt_steps}\n")
        outfile.write("loss per fold\n\n")
        outfile.write("\n".join(str(item) for item in best_losses))
        outfile.write('\n______________________________________________________\n\n')
        outfile.write(f"{met} per fold\n\n")
        outfile.write("\n".join(str(item) for item in best_metric))
        outfile.write('\n______________________________________________________\n\n')
        outfile.write('Mean {} [{:.4} ± {:.4}] IC [{:.4}, {:.4}]\n'.format(met,
            np.mean(best_metric)*100, (np.mean(best_metric) - ic_acc[0])*100, ic_acc[0]*100, ic_acc[1]*100))

        outfile.write('Mean Loss [{:.4} ± {:.4}] IC [{:.4}, {:.4}]\n'.format(
            np.mean(best_losses), (np.mean(best_losses) - ic_loss[0]), ic_loss[0], ic_loss[1]))

        # outfile.write("Confusion matrix\n")
        # outfile.write(str(sum_confusion))


if __name__ == '__main__':

    # TODO add help information to each argument
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--results_path', type=str, default='/shared/sense/sensor_domination/results/nshot_experiments/')
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--ways', required=True, type=int)
    parser.add_argument('--shots', required=True, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--folds', default=0, type=int)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--meta_lr', default=0.006, type=float)
    parser.add_argument('--fast_lr', default=0.03, type=float)
    parser.add_argument('--tune_lr', default=0.002, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--meta_bsz', default=3, type=int)
    parser.add_argument('--train_bsz', default=5, type=int)
    parser.add_argument('--test_bsz', default=5, type=int)
    parser.add_argument('--iterations', default=5000, type=int)
    parser.add_argument('--test_interval', default=100, type=int)
    parser.add_argument('--test_steps', default=120, type=int)
    parser.add_argument('--train_steps', default=50, type=int)

    args = parser.parse_args()

    if args.debug:
        import pydevd_pycharm

        pydevd_pycharm.settrace('172.22.100.7', port=9000, stdoutToServer=True, stderrToServer=True, suspend=False)

    main(args)
