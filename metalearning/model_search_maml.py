import argparse
import logging
import math
import os
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models import ParametrizedMetaSenseModel
from utils.data import TargetDataset
from utils.models import MetaSenseModel
from utils.utils import (fast_adapt_tuning, meta_test, multi_meta_train, multi_metaloaders,
                   save_plot_adapt_steps, balanced_dataset)
import learn2learn as l2l
from torch import nn, optim


def train(model):
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False)

    parameters = list(maml.parameters())
    opt = optim.Adam(parameters, args.meta_lr)

    # Setting scheduler.
    scheduler = torch.optim.lr_scheduler.StepLR(opt, args.iterations // 5, gamma=0.5)
    loss = nn.CrossEntropyLoss(reduction='mean')

    # Outer loop.
    iteration = 0
    while True:
        for meta_loader in meta_loaders.values():
            for batch in meta_loader:

                opt.zero_grad()
                model.zero_grad()

                meta_valid_error = 0.0
                meta_valid_accuracy = 0.0
                accu = 0.0

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
                                                                       args.train_steps,
                                                                       args.shots,
                                                                       args.ways,
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
                                                 test_steps,
                                                 tune_lr,
                                                 device)
                epoch_loss, epoch_acc = min(zip(adapt_values.values()[1], adapt_values.values()[1]))

                with tune.checkpoint_dir(iteration) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'model.pth')
                    torch.save((model.state_dict(), opt.state_dict()), path)

                tune.report(loss=float(epoch_loss), accuracy=epoch_acc.numpy())

                iteration = iteration + 1

                if (iteration + 1) >= args.iterations:
                    break

            if (iteration + 1) >= args.iterations:
                break

        if (iteration + 1) >= args.iterations:
            break

def run_search(config, checkpoint_dir=None, data=None):
    model = MetaSenseModel(3)

    #todo add as configuracoes que vao ser passadas para o modelo


    num_workers = 4 if torch.cuda.is_available() else 0

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

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
        for name, meta_task in data["meta_tasks"].items()
    }
    tune_loader = DataLoader(data["tune_dataset"], batch_size=args.train_bsz, num_workers=num_workers, shuffle=True,
                             pin_memory=True, prefetch_factor=2, drop_last=False)
    test_loader = DataLoader(data["test_dataset"], batch_size=args.test_bsz, num_workers=num_workers, shuffle=False,
                             pin_memory=True, prefetch_factor=2, drop_last=False)

    train(model)


def main(args, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        'num_feat_layers': tune.sample_from(lambda _: np.random.randint(1, 5)),
        'feat_dim': tune.sample_from(lambda _: np.random.randint(2, 9)),
        'kernel_size': tune.choice([3, 5, 7]),
        'clf_hidden_dim': tune.sample_from(lambda _: np.random.randint(2, 9)),
        'lr': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([16, 32, 64, 128, 256]),
    }

    dataset = np.load(args.file_path, allow_pickle=True)
    source_datasets = dataset['Xy_train'][0]

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

    args.fold = 0
    tune_dataset = TargetDataset('tuning', dataset, args.fold, args.shots)


    test_dataset = TargetDataset('test', dataset, args.fold, args.shots)


    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=1e5,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[
            'num_feat_layers',
            'feat_dim',
            'kernel_size',
            'clf_hidden_dim',
            'lr',
            'batch_size'],
        metric_columns=['loss', 'accuracy', 'training_iteration'])

    result = tune.run(
        partial(run_search, data_dir={'train': metaloaders, 'tune': tune_loader, 'test': test_loader}),
        resources_per_trial={'cpu': 2, 'gpu': gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    df = result.results_df
    df.to_csv(os.path.join(args.output_dir, 'results.csv'))

    best_trial = result.get_best_trial(metric='loss', mode='min')
    logging.info('Best trial config: {}'.format(best_trial.config))
    logging.info('Best trial final validation loss: {}'.format(
        best_trial.last_result['loss']))
    logging.info('Best trial final validation accuracy: {}'.format(
        best_trial.last_result['accuracy']))

    best_trained_model = ParametrizedMetaSenseModel(
        best_trial.config['num_feat_layers'],
        best_trial.config['feat_dim'],
        best_trial.config['kernel_size'],
        best_trial.config['clf_hidden_dim'],
        args.num_classes
    ).double()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, 'model.pth'))
    best_trained_model.load_state_dict(model_state)

    test_loader = DataLoader(test_ds, batch_size=best_trial.config['batch_size'], shuffle=False,
                             num_workers=args.num_workers)

    test_acc = test(best_trained_model, test_loader, device)
    logging.info('Best trial test set accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s]: %(message)s')

    parser = argparse.ArgumentParser('Model baselines')

    # General
    parser.add_argument('filepath', type=str, help='Path to `.npz` file.')
    parser.add_argument('-o', '--output-dir', type=str, default='.', help='Output directory.')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Number of training epochs')

    # Misc
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers to use for data-loading')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    numpy_ds = np.load(args.filepath, allow_pickle=True)
    args.num_classes = len(np.unique(numpy_ds['y_test']))

    main(args, num_samples=100, max_num_epochs=10000, gpus_per_trial=1)