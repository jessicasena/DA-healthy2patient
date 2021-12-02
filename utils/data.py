import numpy as np
import torch

from torch.utils import data


def padding(h):
    sizes = np.arange(11)
    ideal_size = 0
    for x in sizes:
        if (2 ** x) >= h:
            ideal_size = 2 ** x
            break
    pad = (ideal_size - h) / 2
    return int(pad)


# Meta-dataset.
class MetaDataset(data.Dataset):

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.labels = np.concatenate([dataset['X_train'], dataset['X_val']],
                                                axis=0), np.concatenate(
            [dataset['y_train'], dataset['y_val']], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()

        sample = self.data[idx].squeeze()
        sample = sample.astype(np.float32)

        pad = padding(sample.shape[-1])
        sample = np.pad(sample, ((0, 0), (pad, pad)), mode='constant')

        target = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return (sample, target)


# Target dataset.
class TargetDataset(data.Dataset):

    def __init__(self, mode, dataset, fold, num_shots):

        self.mode = mode
        self.fold = fold
        self.dataset = dataset
        self.num_shots = num_shots

        # Setting datasets.
        self.make_dataset()

    def make_dataset(self):
        # Making sure the mode is correct.
        assert self.mode in ['train', 'tuning', 'test']
        self.items = []

        self.fewshot_tasks = np.unique(self.dataset['y_test'])

        shot = f'kfold_{self.num_shots}_shot'
        if self.mode == 'train' or self.mode == 'tuning':
            mode_str = 'train_idx'
        elif self.mode == 'test':
            mode_str = 'test_idx'

        # get idx based on num_shots, k-fold and mode
        nshot_idx = self.dataset[shot][self.fold][mode_str]

        # for each task
        for i, t in enumerate(self.fewshot_tasks):
            # get num_shot samples from that task
            for sample, label in zip(self.dataset['X_test'][nshot_idx], self.dataset['y_test'][nshot_idx]):
                if t == label:
                    item = {'acc': sample,
                            'label': i}
                    self.items.append(item)

    def __getitem__(self, index):

        # Reading items from list.
        data = self.items[index]

        acc = np.squeeze(data['acc'])
        lab = data['label']

        # Casting image to the appropriate dtype.
        acc = acc.astype(np.float32)

        # pad goes here
        self.pad = padding(acc.shape[-1])

        acc = np.pad(acc, ((0, 0), (self.pad, self.pad)), mode='constant')

        # Turning to tensors.
        acc = torch.from_numpy(acc)

        return acc, lab

    def __len__(self):

        return len(self.items)


class SensorDataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.padding_size = padding(self.data[0].shape[-1])
        self.classes = np.unique(labels)
        self.class_to_idx = {v: k for k, v in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx].squeeze()
        sample = sample.astype(np.float32)
        sample = np.pad(sample, ((0, 0), (self.padding_size, self.padding_size)), mode='constant')
        target = self.class_to_idx[self.labels[idx]]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(sample)

        return (sample, target)


if __name__ == '__main__':
    fold = 0
    train_shots = 5
    test_shots = 5
    ways = 4
    meta_bsz = 3

    file_path = 'Y:/sensors/nshot_experiments/pra_rodar/f20_t5_4ways_target[pamap2]_source[_mhealth_wharf_wisdm_uschad]_exp1_d_FSL.npz'
    dataset = np.load(file_path, allow_pickle=True)

    # Datasets and dataloaders. dataset, fold, num_shots, num_ways, inner_iters)
    metadataset = MetaDataset(dataset, fold, train_shots, ways, meta_bsz)

    tune_dataset = TargetDataset('tuning', dataset, fold, test_shots)

    test_dataset = TargetDataset('test', dataset, fold, test_shots)

