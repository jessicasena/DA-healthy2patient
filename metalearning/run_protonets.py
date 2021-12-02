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
import torch.nn.functional as F
from learn2learn.data.transforms import KShots, LoadData, NWays, RemapLabels
from torch.utils.data import DataLoader

from utils.data import SensorDataset, TargetDataset
from utils.models import ProtoMetaSenseModel
from utils.utils import (get_metrics, pairwise_distances_logits,
                   save_plot_2lines)


def process_multidatasets(dataset):
    try:
        xx, yy = [], []
        keys = [key for key in dataset['Xy_train'][0].keys()]
        for k in keys:
            train = list(dataset['Xy_train'][0][k])
            train0 = []
            for t in train[0]:
                train0.append(np.squeeze(t))
            xx.extend(train0)
            for name in train[1]:
                yy.append(name.split("-")[1])

        return xx, yy
    except:
        sys.exit("The data format is incorrect")


def main(args):
    start = time.time()
    num_workers = 0
    prefetch_factor = 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder_name = os.path.split(args.file_path)[-1].split('.')[0]
    results_path = os.path.join(args.results_path, folder_name)

    os.makedirs(results_path, exist_ok=True)

    dataset = np.load(args.file_path, allow_pickle=True)
    train_x, train_y = process_multidatasets(dataset)

    train_dataset = SensorDataset(train_x, train_y)
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_ways),
        KShots(train_dataset, args.train_query + args.shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    model = ProtoMetaSenseModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.iterations//5, gamma=0.5)

    best_value = -float('inf')

    model_path = os.path.abspath(os.path.join(results_path, 'best_model.pth'))

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        now = datetime.datetime.now()
        time_now = now.strftime('%d_%m_%Y_%Hh_%Mm_%Ss')

        exp = f'{time_now}'

        results_path = os.path.join(results_path, 'tensorboard', exp)
        writer = SummaryWriter(results_path)
    
    train_f1, ma_f1s = [], []

    for iteration in range(1, args.iterations + 1):
        model.train()

        loss_ctr, n_loss, n_f1 = 0, 0, 0

        for i in range(100):
            batch = next(iter(train_loader))
            evaluation_error, evaluation_f1uracy = fast_adapt(model,
                                                                batch,
                                                                args.train_ways,
                                                                args.shots,
                                                                args.train_query,
                                                                metric=pairwise_distances_logits,
                                                                device=device)

            loss_ctr += 1
            n_loss += evaluation_error.item()
            n_f1 += evaluation_f1uracy['f1-score']

            optimizer.zero_grad()
            evaluation_error.backward()
            optimizer.step()

        scheduler.step()
        interation_f1score = n_f1 / loss_ctr
        interation_loss = n_loss / loss_ctr
        train_f1.append(interation_f1score)
        ma_f1 = np.mean(train_f1[-50:]) if len(train_f1) > 50 else 0
        ma_f1s.append(ma_f1)

        if interation_f1score > best_value:
            best_value = interation_f1score
            torch.save(model.state_dict(), model_path)

        if args.tensorboard:
            writer.add_scalar('Loss/train', interation_loss, iteration)
            writer.add_scalar('Accuracy/train', interation_f1score, iteration)
            writer.add_scalar('Accuracy/train-ma_f1', ma_f1, iteration)


        # Print metrics.
        print(f'Iter {iteration}/{args.iterations} - trn [loss: {interation_loss:.4f} '
            f'- F1 score: {interation_f1score:.4f} - ma_f1: {ma_f1:.4f}]', flush=True)

        # Track accuracy.
        if (iteration + 1) % args.test_interval == 0 or (iteration + 1) == args.iterations:
            # # save plot with test accuracy
            # save_plot(args.test_interval, iteration, test_inner_accuracies, 'test_acc', results_path)
            # save_plot(args.test_interval, iteration, test_inner_errors, 'test_loss', results_path)

            save_plot_2lines(train_f1, ma_f1s, results_path)


    results = {}
    for i in range(5):
        tune_dataset = TargetDataset('tuning', dataset, i, args.shots)
        tune_loader = DataLoader(tune_dataset, batch_size=len(tune_dataset), num_workers=num_workers, shuffle=True,
                                 pin_memory=True, prefetch_factor=prefetch_factor, drop_last=False)
        test_dataset = TargetDataset('test', dataset, i, args.shots)
        test_loader = DataLoader(test_dataset, batch_size=args.test_bsz, num_workers=num_workers, shuffle=False,
                                 pin_memory=True, prefetch_factor=prefetch_factor, drop_last=True)

        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=device))
        model.eval()

        fold_results = {}
        tune_batch = next(iter(tune_loader))
        for step, test_batch in enumerate(test_loader, 1):

            loss, values = fast_adapt_test(model,
                                           args.test_ways,
                                           args.shots,
                                           tune_batch,
                                           test_batch,
                                           metric=pairwise_distances_logits,
                                           device=device)

            for key, value in values.items():
                if key in fold_results:
                    fold_results[key].append(value)
                else:
                    fold_results[key] = [value]

            if 'loss' in fold_results:
                fold_results['loss'].append(loss.item())
            else:
                fold_results['loss'] = [loss.item()]

        for key, value in fold_results.items():
            if key != 'confusion_matrix':
                if key in results:
                    results[key].append(np.mean(value))
                else:
                    results[key] = [np.mean(value)]

    end = time.time()
    duration = datetime.timedelta(seconds=end - start)

    # Save results
    outfile = open(os.path.join(results_path, f'results_protonets_5fold_{args.shots}shot.txt'), 'w', buffering=1)
    outfile.write('Time: {}\n'.format(str(duration)))

    for met, values in results.items():
        if met != 'confusion_matrix':
            ic_acc = st.t.interval(0.9, len(values) - 1, loc=np.mean(values), scale=st.sem(values))

            outfile.write(f'{met} per fold\n')
            outfile.write('\n'.join(str(item) for item in values))
            outfile.write('\n______________________________________________________\n')
            outfile.write('Mean {} [{:.4} ± {:.4}] IC [{:.4}, {:.4}]\n\n'.format(met,
                np.mean(values)*100, (np.mean(values) - ic_acc[0])*100, ic_acc[0]*100, ic_acc[1]*100))


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = metric(query, support)
    loss = F.cross_entropy(logits, labels)
    if device:
        labels = labels.cpu()
        logits = logits.max(1)[1].cpu().detach()

    return loss, get_metrics(logits, labels)


def fast_adapt_test(model, ways, shot, tune_batch, test_batch, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()

    data_tune, labels_tune = tune_batch
    data_test, labels_test = test_batch

    data_tune = data_tune.to(device)
    data_test = data_test.to(device)
    labels_test = labels_test.to(device).long()

    # Compute support and query embeddings
    support = model(data_tune)
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = model(data_test)

    logits = metric(query, support)
    loss = F.cross_entropy(logits, labels_test)
    if device:
        labels_test = labels_test.cpu()
        logits = logits.max(1)[1].cpu().detach()

    return loss, get_metrics(logits, labels_test)


if __name__ == '__main__':

    # TODO add help information to each argument
    parser = argparse.ArgumentParser(description='Prototypical networks for few-shot sensor har.')
    parser.add_argument('--results_path', type=str, default='/shared/sense/sensor_domination/results/nshot_experiments/')
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--train_ways', type=int, default=3)
    parser.add_argument('--test_ways', type=int, default=3)
    parser.add_argument('--shots', required=True, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--folds', default=0, type=int)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--meta_lr', default=0.001, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_bsz', default=5, type=int)
    parser.add_argument('--test_bsz', default=5, type=int)
    parser.add_argument('--iterations', default=500, type=int)
    parser.add_argument('--test_interval', default=100, type=int)
    parser.add_argument('--train_query', type=int, default=15)

    args = parser.parse_args()

    if args.debug:
        import pydevd_pycharm

        pydevd_pycharm.settrace('172.22.100.3', port=9000, stdoutToServer=True, stderrToServer=True, suspend=False)

    main(args)
