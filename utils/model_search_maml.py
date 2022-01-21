import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from functools import partial
import numpy as np
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from data import TargetDataset, SensorDataset
from models import MetaSenseModel
from utils import fast_adapt_tuning, fast_adapt
import learn2learn as l2l
from torch import nn, optim
import copy
from learn2learn.data.transforms import (ConsecutiveLabels, FusedNWaysKShots,
                                         LoadData, RemapLabels)


def train(model, config, args, meta_loaders, tune_loader, test_loader, opt, loss, device, shots, ways, parameters):

    # Setting scheduler.
    scheduler = torch.optim.lr_scheduler.StepLR(opt, args.iterations // 5, gamma=0.5)

    # Outer loop.
    iteration = 0
    while True:
        for meta_loader in meta_loaders.values():
            for batch in meta_loader:

                opt.zero_grad()
                model.zero_grad()

                # Inner loop.
                for task in range(args.meta_bsz):
                    opt.zero_grad()
                    model.zero_grad()
                    model.train()
                    # Compute meta-training loss
                    learner = model.clone()
                    evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                       task,
                                                                       learner,
                                                                       loss,
                                                                       config["train_steps"],
                                                                       shots,
                                                                       ways,
                                                                       device)
                    model.zero_grad()
                    evaluation_error.backward()

                # Average the accumulated gradients and optimize.
                for p in parameters:
                    p.grad.data.mul_(1.0 / args.meta_bsz)

                opt.step()

                # Take LR scheduler step.
                scheduler.step()

                # Compute meta-testing loss.
                learner = copy.deepcopy(model)
                adapt_values = fast_adapt_tuning(tune_loader,
                                                 test_loader,
                                                 learner,
                                                 loss,
                                                 args.test_steps,
                                                 config["tune_lr"],
                                                 device)

                acc = float('inf')
                min_loss = float('inf')
                for key, value in adapt_values.items():
                    if value[1] < min_loss:
                        min_loss = value[1]
                        acc = value[0]['accuracy']
                epoch_loss, epoch_acc = min_loss, acc
                print(f'Epoch {iteration} Loss: {epoch_loss} Acc: {epoch_acc}', flush=True)

                with tune.checkpoint_dir(iteration) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'model.pth')
                    torch.save((model.state_dict(), opt.state_dict()), path)

                tune.report(loss=float(epoch_loss), accuracy=epoch_acc)

                iteration = iteration + 1

                if (iteration + 1) >= args.iterations:
                    break

            if (iteration + 1) >= args.iterations:
                break

        if (iteration + 1) >= args.iterations:
            break


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def run_search(config, checkpoint_dir=None, dataset=None):
    args = dataset["args"]

    dataset = np.load(args.file_path, allow_pickle=True)
    source_datasets = dataset['Xy_train'][0]

    args.fold = 0
    tune_dataset = TargetDataset('tuning', dataset, args.fold, args.shots)
    test_dataset = TargetDataset('test', dataset, args.fold, args.shots)

    # Datasets and dataloaders.
    metadatasets = {name: SensorDataset(data, labels) for name, (data, labels) in source_datasets.items()}
    metadatasets = {name: l2l.data.MetaDataset(metadataset) for name, metadataset in metadatasets.items()}

    # Meta-Train transform and set.
    meta_transforms = {
        name: [
            FusedNWaysKShots(metadataset, n=args.ways, k=2 * args.shots),
            LoadData(metadataset),
            RemapLabels(metadataset),
            ConsecutiveLabels(metadataset),
        ] for name, metadataset in metadatasets.items()
    }
    meta_tasks = {
        name: l2l.data.TaskDataset(metadataset,
                                   task_transforms=meta_transforms[name],
                                   num_tasks=10000 * args.meta_bsz)
        for name, metadataset in metadatasets.items()
    }
    model = MetaSenseModel(3)

    model = l2l.algorithms.MAML(model, lr=config["fast_lr"], first_order=False)
    parameters = list(model.parameters())
    num_workers = 4 if torch.cuda.is_available() else 0

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = optim.Adam(parameters, config["meta_lr"])
    loss = nn.CrossEntropyLoss(reduction='mean')

    if args.checkpoint:
        model_state, optimizer_state = torch.load(
            os.path.join(args.checkpoint, 'model.pt'),
            map_location=device
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    metaloaders = {
        name: DataLoader(meta_task, batch_size=args.meta_bsz, num_workers=num_workers, pin_memory=True,
                         prefetch_factor=2)
        for name, meta_task in meta_tasks.items()
    }
    tune_loader = DataLoader(tune_dataset, batch_size=args.train_bsz, num_workers=num_workers, shuffle=True,
                             pin_memory=True, prefetch_factor=2, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_bsz, num_workers=num_workers, shuffle=False,
                             pin_memory=True, prefetch_factor=2, drop_last=False)

    train(model, config, args, metaloaders, tune_loader, test_loader, optimizer, loss, device, args.shots, args.ways, parameters)


def main(args, num_samples=10):
    config = {
        'fast_lr': tune.loguniform(1e-4, 1e-1),
        'meta_lr': tune.loguniform(1e-4, 1e-1),
        'tune_lr': tune.loguniform(1e-4, 1e-1),
        'train_steps': tune.choice([5, 10, 20, 30, 40, 50, 60]),
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=1e5,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[
            'fast_lr',
            'meta_lr',
            'tune_lr',
            'train_steps'],
        metric_columns=['loss', 'accuracy', 'training_iteration'])

    os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
    result = tune.run(
        partial(run_search, dataset={"args":args}),
        resources_per_trial={'cpu': 1, 'gpu': 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    df = result.results_df
    df.to_csv(os.path.join(args.results_path, 'results.csv'))

    best_trial = result.get_best_trial(metric='loss', mode='min')
    logging.info('Best trial config: {}'.format(best_trial.config))
    logging.info('Best trial final validation loss: {}'.format(
        best_trial.last_result['loss']))
    logging.info('Best trial final validation accuracy: {}'.format(
        best_trial.last_result['accuracy']))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s]: %(message)s')

    parser = argparse.ArgumentParser('Model baselines')

    # General
    parser.add_argument('--results_path', type=str,
                        default='/shared/sense/sensor_domination/results/nshot_experiments/')
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--ways', default=3, type=int)
    parser.add_argument('--shots', default=1,  type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--meta_bsz', default=3, type=int)
    parser.add_argument('--train_bsz', default=5, type=int)
    parser.add_argument('--test_bsz', default=5, type=int)
    parser.add_argument('--iterations', default=5000, type=int)
    parser.add_argument('--test_interval', default=100, type=int)
    parser.add_argument('--test_steps', default=120, type=int)
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Path to checkpoint file')

    args = parser.parse_args()

    numpy_ds = np.load(args.file_path, allow_pickle=True)
    args.num_classes = len(np.unique(numpy_ds['y_test']))

    main(args, num_samples=100)
