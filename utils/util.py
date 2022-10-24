import copy
import math
import os
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix, precision_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.data import SensorDataset
import learn2learn as l2l
from torch.utils.data import DataLoader
from learn2learn.data.transforms import (ConsecutiveLabels, FusedNWaysKShots,
                                         LoadData, RemapLabels)
import torchvision
from time import time
import datetime
from sklearn.metrics import roc_curve
from scipy import stats as st
import gc
import optuna
# Timing utilities
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1-score': f1_score(y_true, y_pred, average=None, labels=[0,1]),
        'f1-score_macro': f1_score(y_true, y_pred, average="macro", labels=[0, 1]),
        'recall': recall_score(y_true, y_pred, average=None, zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average="macro", zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'confusion_matrix_norm_true': confusion_matrix(y_true, y_pred, normalize='true'),
        'precision': precision_score(y_true, y_pred, average=None, zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average="macro", zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred, average="macro")
    }


def print_metrics(logger, n_classes, cum_acc, cum_recall, cum_precision, cum_auc, cum_f1, cum_recall_macro, cum_precision_macro,
                  cum_f1_macro):
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

    logger.info('accuracy: {:.2f} ± {:.2f}\n'.format(np.mean(current_acc) * 100,
                                                          abs(np.mean(current_acc) - ci_mean[0]) * 100))

    logger.info('recall_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall_macro) * 100,
                                                              abs(np.mean(current_recall_macro) - ci_recall_macro[
                                                                  0]) * 100))
    logger.info('precision_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec_macro) * 100,
                                                                 abs(np.mean(current_prec_macro) - ci_prec_macro[
                                                                     0]) * 100))
    logger.info('f1-score_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1_macro) * 100,
                                                                abs(np.mean(current_f1_macro) - ci_f1_macro[
                                                                    0]) * 100))
    logger.info('roc_auc: {:.2f} ± {:.2f}\n'.format(np.mean(current_auc) * 100,
                                                         abs(np.mean(current_auc) - ci_auc[0]) * 100))

    for class_ in range(n_classes):
        logger.info(f"Class: {class_}")

        current_f1 = np.array(cum_f1)[:, class_]
        current_recall = np.array(cum_recall)[:, class_]
        current_prec = np.array(cum_precision)[:, class_]

        ci_f1 = st.t.interval(0.95, len(current_f1) - 1, loc=np.mean(current_f1), scale=st.sem(current_f1))
        ci_recall = st.t.interval(0.95, len(current_recall) - 1, loc=np.mean(current_recall),
                                  scale=st.sem(current_recall))
        ci_prec = st.t.interval(0.95, len(current_prec) - 1, loc=np.mean(current_prec), scale=st.sem(current_prec))

        logger.info('recall: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall) * 100,
                                                            abs(np.mean(current_recall) - ci_recall[0]) * 100))
        logger.info('precision: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec) * 100,
                                                               abs(np.mean(current_prec) - ci_prec[0]) * 100))
        logger.info('f1-score: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1) * 100,
                                                         abs(np.mean(current_f1) - ci_f1[0]) * 100))

            #
# def balanced_dataset(X, y):
#     from imblearn.under_sampling import RandomUnderSampler
#     from collections import Counter
#     under_sampler = RandomUnderSampler(random_state=42)
#     X_res, y_res = under_sampler.fit_resample(np.array(X).reshape(-1, 1), np.array(y).reshape(-1, 1))
#     print(f"Statistics before balancing: {Counter(y)}", flush=True)
#     print(f"Statistics after balancing: {Counter(y_res)}", flush=True)
#
#     return X_res.squeeze(), y_res.squeeze()


def validation(model, loader, device, criterion, focal_loss=False, use_cuda=True):
    model.eval()
    y_true = []
    y_pred = []
    loss_total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)

            loss = criterion(preds, targets)

            loss_total += loss.item()

            if use_cuda:
                targets = targets.cpu()

            preds = torch.argmax(preds, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    return loss_total / len(loader), get_metrics(y_true, y_pred)


def train(model, train_loader, early_stopping, optimizer, criterion, device, scheduler, best_model_folder, writer, trial, focal_loss= False, epochs=100, use_cuda=True, use_additional_data=False):
    step = 0
    start_timer()
    best_metric = 0
    # Constructs scaler once, at the beginning of the convergence run, using default args.
    # If your network fails to converge with default GradScaler args, please file an issue.
    # The same GradScaler instance should be used for the entire convergence run.
    # If you perform multiple convergence runs in the same script, each run should use
    # a dedicated fresh GradScaler instance.  GradScaler instances are lightweight.
    scaler = torch.cuda.amp.GradScaler()
    #for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        start = time()
        model.train()

        for acc, add_data, targets in train_loader:
            # print class distribution
            # c, n = np.unique(targets, return_counts=True)
            # if len(n) > 1:
            #     print(f"\nClass {n[0]/len(targets)*100}/{n[1]/len(targets)*100}", flush=True)
            # else:
            #     print(f"\nClass {c[0]} {n[0]/len(targets)*100}", flush=True)
            # Runs the forward pass under autocast.
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if use_cuda:
                    acc, targets = acc.to(device), targets.to(device)
                    if use_additional_data:
                        add_data = add_data.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if use_additional_data:
                    preds = model(acc, add_data)
                else:
                    preds = model(acc)

                # output is float16 because linear layers autocast to float16.
                assert preds.dtype is torch.float16
                # y_true = np.argmax(targets.detach().cpu(), axis=1)
                # y_pred = np.argmax(preds.detach().cpu(), axis=1)
                # print(f'preds = {y_pred}')
                # print(f'target = {y_true}\n')

                if focal_loss:
                    loss = criterion(preds, targets)
                else:
                    loss = criterion(preds, targets)

                # loss is float32 because mse_loss layers autocast to float32.
                assert loss.dtype is torch.float32
            # Exits autocast before backward().
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            optimizer.zero_grad()  # set_to_none=True here can modestly improve performance

            # calculate metrics and print statistics
            # if use_cuda:
            #     targets = targets.cpu()
            # y_true = targets.numpy()
            # y_pred = [1 if x > 0 else 0 for x in preds.detach().cpu().numpy()]
            # metrics = get_metrics(y_true, y_pred)

            step += 1

        scheduler.step()
        # early stopping
        current_loss, current_metric = validation(model, early_stopping["val_loader"], device, criterion, focal_loss, use_cuda, use_additional_data)
        end = time()
        epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(epochs))).format(epoch + 1)
        epoch_fin = 'loss: {:.6f}, F1: {} [{}]'.format(current_loss, current_metric['f1-score_macro'], str(datetime.timedelta(seconds=(end - start))))
        print(epoch_desc + epoch_fin, flush=True)

        writer.add_scalar('Loss/train', current_loss, epoch)
        writer.add_scalar('Metrics/f1_macro', current_metric['f1-score_macro'], epoch)

        if current_loss > early_stopping["last_loss"]:
            early_stopping["trigger_times"] += 1
            if early_stopping["trigger_times"] >= early_stopping["patience"]:
                print('Early stopping!\n', flush=True)
                return model
        else:
            early_stopping["trigger_times"] = 0

        if current_metric['f1-score_macro'] > best_metric:
            best_metric = current_metric['f1-score_macro']
            checkpoint = {"model": model.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict()}
            torch.save(checkpoint, best_model_folder)
            print(f"Best model. loss: {current_loss}, F1 macro: {current_metric['f1-score_macro']}", flush=True)

        early_stopping["last_loss"] = current_loss
        if trial is not None:
            trial.report(current_metric['f1-score_macro'], epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    end_timer_and_print(f"Training session ({epochs}epochs)")


def test(model, loader, device, use_cuda=True, use_additional_data=False):
    print('Eval model...')
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        #for inputs, add_data, targets in tqdm(loader):
        for inputs, add_data, targets in loader:
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
                if use_additional_data:
                    add_data = add_data.to(device)

            if use_additional_data:
                preds = model(inputs, add_data)
            else:
                preds = model(inputs)

            if use_cuda:
                targets = targets.cpu()

            y_true.extend(targets)
            y_pred.extend(np.argmax(preds.detach().cpu(), axis=1))

    return get_metrics(y_true, y_pred)


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

    learner.train()
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
            learner.zero_grad()

            prd, _ = learner(acc)

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
        predictions, _ = learner(adaptation_data)
        train_error = loss(predictions, adaptation_labels)
        learner.adapt(train_error)

    predictions, _ = learner(evaluation_data)
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

            prd, _ = learner(acc)

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