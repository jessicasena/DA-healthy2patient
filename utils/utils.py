import copy
import math
import os
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix
from tqdm import tqdm
from .data import SensorDataset
import learn2learn as l2l
from torch.utils.data import DataLoader
from learn2learn.data.transforms import (ConsecutiveLabels, FusedNWaysKShots,
                                         LoadData, RemapLabels)


def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1-score': f1_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def train(model, loader, optimizer, criterion, epochs=100, use_cuda=True):
    for epoch in range(epochs):
        with tqdm(loader) as pbar:
            epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(epochs))).format(epoch + 1)
            pbar.set_description(epoch_desc)
            for inputs, targets in pbar:
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                preds = model(inputs)
                loss = criterion(preds, targets.long())

                if use_cuda:
                    targets = targets.cpu()
                    
                y_true = targets.numpy()
                y_pred = np.argmax(preds.detach().cpu().numpy(), axis=1)

                metrics = get_metrics(y_true, y_pred)

                pbar.set_postfix(loss='{0:.6f}'.format(loss), accuracy='{0:.04f}'.format(metrics['accuracy']))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def test(model, loader, use_cuda=True):
    model.eval()
    print('Eval model...')
    y_true = []
    y_pred = []
    for inputs, targets in tqdm(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        preds = model(inputs)

        if use_cuda:
            targets = targets.cpu()
            
        y_true.extend(targets.numpy())
        y_pred.extend(np.argmax(preds.detach().cpu().numpy(), axis=1))

    return get_metrics(y_true, y_pred)


def save_plot(test_interval, iteration, infos, title, results_path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(range(test_interval, (iteration + 1) + test_interval, test_interval),
            np.asarray(infos))
    ax.set_title(title)

    plt.savefig(os.path.join(results_path, f'{title}.pdf'))
    plt.close(fig)


def save_plot_2lines(acc, ma_acc, results_path):

    plt.plot(np.asarray(acc), label="acc")

    plt.plot(np.asarray(ma_acc), label="ma_acc")

    plt.title('Train accuracy and mean average accuracy')

    plt.legend()

    plt.savefig(os.path.join(results_path, 'train_acc.pdf'))
    plt.close()


def save_plot_adapt_steps(adapt_values, accs, results_path):

    plt.plot(adapt_values, accs)

    plt.title('Accuracy x Tuning adaptation steps (mean 5 folds)')

    plt.savefig(os.path.join(results_path, 'adapt_steps.pdf'))
    plt.close()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def meta_train(model, device, loss, results_path, meta_loader, opt, scheduler, parameters, args):
    best_value = - float("inf")

    model_path = os.path.abspath(os.path.join(results_path, 'best_model.th'))

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        now = datetime.now()
        time_now = now.strftime("%d_%m_%Y_%Hh_%Mm_%Ss")

        exp = f'mini_mlr{args.meta_lr}_flr{args.fast_lr}_metab{args.meta_bsz}_tststep{args.test_steps}_trtstep{args.train_steps}_{time_now}'

        results_path = os.path.join(results_path, "tensorboard", exp)
        writer = SummaryWriter(results_path)


    # Presetting lists.
    train_inner_errors = []
    train_inner_accuracies = []
    ma_accs = []

    # Outer loop.
    iteration = 0
    while True:

        for batch in meta_loader:

            opt.zero_grad()

            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            accu = 0.0

            # Inner loop.
            for task in range(args.meta_bsz):
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
                evaluation_error.backward()
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            if meta_valid_accuracy / args.meta_bsz > best_value:
                best_value = meta_valid_accuracy / args.meta_bsz
                with open(model_path, 'wb') as f:
                    torch.save(model.state_dict(), f)

            train_inner_errors.append(meta_valid_error / args.meta_bsz)
            train_inner_accuracies.append(meta_valid_accuracy / args.meta_bsz)

            ma_acc = np.mean(train_inner_accuracies[-50:]) if len(train_inner_accuracies) > 50 else 0
            ma_accs.append(ma_acc)

            if args.tensorboard:
                writer.add_scalar("Loss/train", meta_valid_error / args.meta_bsz, iteration)
                writer.add_scalar("Accuracy/train", meta_valid_accuracy / args.meta_bsz, iteration)
                writer.add_scalar("Accuracy/train-ma_acc", ma_acc, iteration)


            # Print metrics.
            print(f'Iter {iteration + 1}/{args.iterations} - trn [loss: {meta_valid_error / args.meta_bsz:.4f} '
                  f'- acc: {meta_valid_accuracy / args.meta_bsz:.4f} - ma_acc: {ma_acc:.4f}]', flush=True)

            # Track accuracy.
            if (iteration + 1) % args.test_interval == 0 or (iteration + 1) == args.iterations:
                # # save plot with test accuracy
                # save_plot(args.test_interval, iteration, test_inner_accuracies, 'test_acc', results_path)
                # save_plot(args.test_interval, iteration, test_inner_errors, 'test_loss', results_path)

                save_plot_2lines(train_inner_accuracies, ma_accs, results_path)

            # Average the accumulated gradients and optimize.
            for p in parameters:
                p.grad.data.mul_(1.0 / args.meta_bsz)

            opt.step()

            # Take LR scheduler step.
            scheduler.step()

            iteration = iteration + 1

            if (iteration + 1) >= args.iterations:
                break

        if (iteration + 1) >= args.iterations:
            break


def multi_meta_train(model, device, loss, results_path, meta_loaders, opt, scheduler, parameters, args):
    best_value = - float("inf")

    model_path = os.path.abspath(os.path.join(results_path, 'best_model.th'))

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        now = datetime.now()
        time_now = now.strftime("%d_%m_%Y_%Hh_%Mm_%Ss")

        exp = f'mini_mlr{args.meta_lr}_flr{args.fast_lr}_metab{args.meta_bsz}_tststep{args.test_steps}_trtstep{args.train_steps}_{time_now}'

        results_path = os.path.join(results_path, "tensorboard", exp)
        writer = SummaryWriter(results_path)


    # Presetting lists.
    train_inner_errors = []
    train_inner_accuracies = []
    ma_accs = []

    # Outer loop.
    iteration = 0
    while True:
        for meta_loader in meta_loaders.values():
            for batch in meta_loader:

                opt.zero_grad()

                meta_valid_error = 0.0
                meta_valid_accuracy = 0.0
                accu = 0.0

                # Inner loop.
                for task in range(args.meta_bsz):
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
                    evaluation_error.backward()
                    meta_valid_error += evaluation_error.item()
                    meta_valid_accuracy += evaluation_accuracy.item()

                if meta_valid_accuracy / args.meta_bsz > best_value:
                    best_value = meta_valid_accuracy / args.meta_bsz
                    with open(model_path, 'wb') as f:
                        torch.save(model.state_dict(), f)

                train_inner_errors.append(meta_valid_error / args.meta_bsz)
                train_inner_accuracies.append(meta_valid_accuracy / args.meta_bsz)

                ma_acc = np.mean(train_inner_accuracies[-50:]) if len(train_inner_accuracies) > 50 else 0
                ma_accs.append(ma_acc)

                if args.tensorboard:
                    writer.add_scalar("Loss/train", meta_valid_error / args.meta_bsz, iteration)
                    writer.add_scalar("Accuracy/train", meta_valid_accuracy / args.meta_bsz, iteration)
                    writer.add_scalar("Accuracy/train-ma_acc", ma_acc, iteration)


                # Print metrics.
                print(f'Iter {iteration + 1}/{args.iterations} - trn [loss: {meta_valid_error / args.meta_bsz:.4f} '
                    f'- acc: {meta_valid_accuracy / args.meta_bsz:.4f} - ma_acc: {ma_acc:.4f}]', flush=True)

                # Track accuracy.
                if (iteration + 1) % args.test_interval == 0 or (iteration + 1) == args.iterations:
                    # # save plot with test accuracy
                    # save_plot(args.test_interval, iteration, test_inner_accuracies, 'test_acc', results_path)
                    # save_plot(args.test_interval, iteration, test_inner_errors, 'test_loss', results_path)

                    save_plot_2lines(train_inner_accuracies, ma_accs, results_path)

                # Average the accumulated gradients and optimize.
                for p in parameters:
                    p.grad.data.mul_(1.0 / args.meta_bsz)

                opt.step()

                # Take LR scheduler step.
                scheduler.step()

                iteration = iteration + 1

                if (iteration + 1) >= args.iterations:
                    break

            if (iteration + 1) >= args.iterations:
                break

        if (iteration + 1) >= args.iterations:
            break


def multi_metaloaders(source_datasets, num_workers, prefetch_factor, args):
    # Datasets and dataloaders.
    metadatasets = {name: SensorDataset(data, labels) for name, (data, labels) in source_datasets.items()}
    metadatasets = {name: l2l.data.MetaDataset(metadataset) for name, metadataset in metadatasets.items()}

    # Meta-Train transform and set.
    meta_transforms = {
        name: [
            FusedNWaysKShots(metadataset, n=args.ways, k=2*args.shots),
            LoadData(metadataset),
            RemapLabels(metadataset),
            ConsecutiveLabels(metadataset),
        ] for name, metadataset in metadatasets.items()
    }
    meta_tasks = {
        name: l2l.data.TaskDataset(metadataset,
                                   task_transforms=meta_transforms[name],
                                   num_tasks=10000*args.meta_bsz)
        for name, metadataset in metadatasets.items()
    }

    metaloaders = {
        name: DataLoader(meta_task, batch_size=args.meta_bsz, num_workers=num_workers, pin_memory=True,
                         prefetch_factor=prefetch_factor)
        for name, meta_task in meta_tasks.items()
    }

    return metaloaders


def meta_test(model, results_path, tune_loader, test_loader, loss, test_steps, tune_lr, device):
    model_path = os.path.abspath(os.path.join(results_path, 'best_model.th'))

    print("Testing", flush=True)
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    # Compute meta-testing loss.
    learner = copy.deepcopy(model)
    adapt_values = fast_adapt_tuning(tune_loader,
                                     test_loader,
                                     learner,
                                     loss,
                                     test_steps,
                                     tune_lr,
                                     device)
    return adapt_values


def fast_adapt_tuning(
        tune_loader,
        test_loader,
        learner,
        loss,
        adaptation_steps,
        tune_lr,
        device,
        adapt_test_interval=5):



    # Setting tuning optimizer.
    adapt_opt = torch.optim.Adam(list(learner.parameters()),
                                 lr=tune_lr,
                                 betas=(0, 0.999))

    # Setting tuning scheduler.
    adapt_scheduler = torch.optim.lr_scheduler.StepLR(adapt_opt,
                                                      adaptation_steps // 5,
                                                      gamma=0.5)
    err = 0
    adapt_values = {}
    # Adapt the model.
    for step in range(adaptation_steps):

        for batch in tune_loader:

            acc, lab = batch

            acc = acc.to(device)
            lab = lab.to(device)

            adapt_opt.zero_grad()

            prd = learner(acc)

            err = loss(prd, lab)

            err.backward()

            adapt_opt.step()

        adapt_scheduler.step()

        if (step + 1) % adapt_test_interval == 0:
            valid_acc = tuning_test(learner, test_loader, device)
            adapt_values[step + 1] = [valid_acc, err.item()]

    return adapt_values


def fast_adapt(batch,
               task,
               learner,
               loss,
               adaptation_steps,
               shots,
               ways,
               device=None):

    data, labels = batch
    data, labels = data[task].to(device).float(), labels[task].squeeze(0).to(device)

    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for _ in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = balanced_accuracy_score(predictions.max(1)[1].cpu().numpy(), evaluation_labels.cpu().numpy())
    return valid_error, valid_accuracy


def tuning_test(learner, test_loader, device):
    with torch.no_grad():

        learner.eval()

        # Evaluate the adapted model.
        prd_list = []
        lab_list = []

        for batch in test_loader:

            acc, lab = batch

            acc = acc.to(device)
            lab = lab.to(device)

            prd = learner(acc)

            prd_list.extend(prd.max(1)[1].cpu().numpy().tolist())
            lab_list.extend(lab.cpu().numpy().tolist())

        prd_np = np.asarray(prd_list)
        lab_np = np.asarray(lab_list)

        valid_acc = get_metrics(lab_np, prd_np)
    learner.train()

    return valid_acc


def pairwise_distances_logits(a, b):
    """
    Compute the pairwise distance matrix between the vectors in a and b
    using the pairwise distances between the logits of each vector.
    d = -||f(x) - c_k||^2 = ||f(x)||^2 - 2 * f(x) * c_k + c_k^2
    """
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits