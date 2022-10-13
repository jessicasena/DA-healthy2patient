import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from utils.models import MetaSenseModel
from torch.utils.data import DataLoader
from utils.util import train
from utils.data import SensorDataset
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

def extract_activations(model, dataloader, use_cuda, max_samples=1000):
    """
    Iterate though each (subset of) dataset and store activations.

    Parameters:
        model (torch.model): the model to evaluate, output representation of input image with size D
        dataloader (torch.utils.data.DataLoader): Dataloader, with batch size 1
        max_samples (int): number of samples to evaluate, N

    Returns:
        activations (numpy.array): Array of size NxD
    """
    model.eval()

    activations = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = batch[0]
            if idx >= max_samples:
                break
            print(f'\r{idx}/{min(len(dataloader), max_samples)}', end="")
            if use_cuda:
                batch = batch.cuda()
            _, out = model(batch)

            if use_cuda:
                out = out.cpu()
            activations.extend(out.numpy())

    return np.asarray(activations)


def representation_shift(act_ref, act_test):
    """
    Calculate representation shift using Wasserstein distance

    Parameters:
        act_ref (numpy.array): Array of size NxD
        act_test (numpy.array): Array of size NxD

    Returns:
        representation_shift (float): Mean Wasserstein distance over all channels (D)
    """
    print("Calculate representation shift\n", flush=True)
    wass_dist = [wasserstein_distance(act_ref[:, channel], act_test[:, channel]) for channel in range(act_ref.shape[1])]
    return np.asarray(wass_dist).mean()


def train_model(args, out_loader, model):
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.use_cuda:
        model = model.cuda()

    train(model, out_loader, optim, criterion, epochs=args.num_epochs, use_cuda=args.use_cuda)
    outpath = os.path.join(args.output, args.filepath.split(os.sep)[-1][:-4])
    checkpoint_filepath = outpath + '.pth'
    with open(checkpoint_filepath, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(checkpoint_filepath, n_classes):
    model = MetaSenseModel(n_classes)
    model.load_state_dict(torch.load(checkpoint_filepath))
    return model


def get_model(args, dataloader, n_classes):
    device = torch.device('cuda' if args.use_cuda
                                    and torch.cuda.is_available() else 'cpu')

    outpath = os.path.join(args.output, args.filepath.split(os.sep)[-1][:-4])
    checkpoint_filepath = outpath + '.pth'

    model = MetaSenseModel(n_classes)
    if args.train:
        print("Train model ...", flush=True)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        if args.use_cuda:
            model = model.cuda()

        train(model, dataloader, optim, criterion, epochs=args.num_epochs, use_cuda=args.use_cuda)

        with open(checkpoint_filepath, 'wb') as f:
            torch.save(model.state_dict(), f)
        print("Done", flush=True)
    else:
        # Reload model weights
        with open(checkpoint_filepath, 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=device))

    return model


def calc_a_distance(source_train, target_train, source_test, target_test):
    print("Calculate a-distance\n", flush=True)
    X_train = np.concatenate((source_train, target_train), axis=0)
    y_train = np.concatenate((np.zeros(source_train.shape[0]), np.ones(target_train.shape[0])), axis=0)

    X_test = np.concatenate((source_test, target_test), axis=0)
    y_test = np.concatenate((np.zeros(source_test.shape[0]), np.ones(target_test.shape[0])), axis=0)

    model = LinearSVC(verbose=True, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_hat)

    a_distance = 2. * (1. - 2. * mae)

    return a_distance


def get_dataloader(args, data, labels):
    dataset = SensorDataset(data, labels)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=True,
                                  shuffle=True)

    return dataloader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model baselines')

    # General
    parser.add_argument('filepath', type=str, help='Path to `.npz` file.')
    parser.add_argument('-o', '--output', type=str, default='./', help='Output path folder')

    # Optimization
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Number of instances per batch')
    parser.add_argument('-l', '--learning-rate', type=int, default=1e-3, help='Epoch learning rate')
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help='Number of outing epochs')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers to use for data-loading')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    args.num_ways = 3

    args.use_cuda = torch.cuda.is_available()

    os.makedirs(args.output, exist_ok=True)

    out_file = os.path.join(args.output, os.path.split(args.filepath)[-1][:-4])
    out_file = out_file + '_results.txt'

    f = open(out_file, 'w')
    repr_shift = []
    adistance = []
    repr_shift.append('Representation Shift\n')
    adistance.append('A-distance\n')

    dataset = np.load(args.filepath, allow_pickle=True)

    out_datasets = list(dataset['Xy_train'][0].keys())
    fold = dataset['kfold_no_shot'][0]

    reference_data, reference_labels = dataset['X_test'][fold['train_idx']].squeeze(), dataset['y_test'][
        fold['train_idx']]
    in_data, in_labels = dataset['X_test'][fold['test_idx']].squeeze(), dataset['y_test'][fold['test_idx']]
    labels2idx = {k: idx for idx, k in enumerate(np.unique(dataset['y_test']))}
    reference_labels = np.array([labels2idx[label] for label in reference_labels])
    in_labels = np.array([labels2idx[label] for label in in_labels])

    reference_loader = get_dataloader(args, reference_data, reference_labels)

    model = get_model(args, reference_loader, len(reference_labels))

    print("Extracting activations - reference", flush=True)
    features_ref = extract_activations(model, reference_loader, args.use_cuda)

    in_loader = get_dataloader(args, in_data, in_labels)

    print("Extracting activations - in dataset", flush=True)
    features_in = extract_activations(model, in_loader, args.use_cuda)
    wass_in_distance = representation_shift(features_ref, features_in)

    repr_shift.append('In-distribution distance: {}\n'.format(wass_in_distance))

    repr_shift.append('Out-of-distribution distance:\n')

    for out_name in tqdm(out_datasets):
        out_data, out_labels = dataset['Xy_train'][0][out_name]
        out_data, out_labels = np.array(out_data).squeeze(), np.array(out_labels)
        out_labels2idx = {k: idx for idx, k in enumerate(np.unique(out_labels))}
        out_labels = np.array([out_labels2idx[label] for label in out_labels])

        out_X_train, out_X_test, out_y_train, out_y_test = train_test_split(out_data, out_labels, test_size = 0.20, random_state = 42)

        out_train_loader = get_dataloader(args, out_X_train, out_y_train)
        out_test_loader = get_dataloader(args, out_X_test, out_y_test)

        print(f"Extracting activations - out  ({out_name})\n", flush=True)
        features_out_train = extract_activations(model, out_train_loader, args.use_cuda)
        features_out_test = extract_activations(model, out_test_loader, args.use_cuda)
        wass_out_distance = representation_shift(features_ref, features_out_test)

        repr_shift.append(f'{out_name}: {wass_out_distance}\n')

        a_dist = calc_a_distance(features_ref, features_in, features_out_train, features_out_test)
        adistance.append(f'{out_name}: {a_dist}\n')

    for line in repr_shift:
        f.write(line)
    for line in adistance:
        f.write(line)

    f.close()