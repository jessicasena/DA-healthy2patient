import datetime
import sys
from time import time

from sklearn.utils import class_weight

sys.path.append("/home/jsenadesouza/DA-healthy2patient/code/")

import math
import random
import torch
import numpy as np
import json
import logging
from models.IMU_Transformer.util import utils
from os.path import join
from models.IMU_Transformer.trans_models.IMUTransformer import IMUTransformerEncoder
from models.IMU_Transformer.trans_models.IMUCLSBaseline import IMUCLSBaseline
from utils.util import get_metrics, print_metrics, start_timer, end_timer_and_print, validation
from models.util import load_data, set_logger, split_data, get_loaders
import torchtest as tt


def magnitude(sample):
    mag_vector = []
    for s in sample:
        mag_vector.append(math.sqrt(sum([s[0] ** 2, s[1] ** 2, s[2] ** 2])))
    return mag_vector


def test_network(model, loss, optim, batch, device):
    # run all tests
    print(tt.test_suite(
        model,
        loss,  # loss function
        optim,  # optimizer
        batch,  # random data
        test_output_range=False,
        test_vars_change=True,
        test_nan_vals=True,
        test_inf_vals=True,
        test_gpu_available=True,
        device=device
    ))

if __name__ == "__main__":
    utils.init_logger()

    # Record execution details
    logging.info("Starting IMU-transformers")

    # Read configuration
    with open('/home/jsenadesouza/DA-healthy2patient/code/models/IMU_Transformer/config.json', "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    #data_path = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/t900_INTELLIGENT_PAIN_ADAPT_15min.npz"
    data_path = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/PAIN_15min.npz"
    checkpoint_path = None
    #checkpoint_path = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/IMUTransformers/out/run_10_10_22_22_20_final.pth"
    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    #device_id = 'cpu'
    torch_seed = 42
    numpy_seed = 42
    torch.manual_seed(torch_seed)
    device_id = None
    if use_cuda:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    if config.get("use_baseline"):
        model = IMUCLSBaseline(config).to(device)
    else:
        model = IMUTransformerEncoder(config).to(device)

    # Set the dataset and data loader
    logging.info("Start train data preparation")
    # read data and get dataset and dataloader
    # split the data into train, val and test sets
    # Split data
    X, y, y_target = load_data(data_path, clin_variable_target="pain_score_class")
    labels2idx = {k: idx for idx, k in enumerate(np.unique(y_target))}

    patients = list(np.unique(y[:, -1]))
    random.shuffle(patients)
    patient_splits = np.array_split(patients, 6)
    num_folders = 5
    n_classes = 2
    cum_acc, cum_recall, cum_precision, cum_auc, cum_f1 = [], [], [], [], []
    cum_recall_macro, cum_precision_macro, cum_f1_macro = [], [], []
    for folder_idx in range(num_folders):
        train_data, train_labels, test_data, test_labels, val_data, val_labels = split_data(X, y, y_target, labels2idx,
                                                                                            logging, patient_splits,
                                                                                            folder_idx)
        # get dataloaders
        train_loader, test_loader, val_loader = get_loaders(config.get("batch_size"), train_data, train_labels, test_data, test_labels,
                                                            val_data, val_labels, weighted_sampler=False)

        # zero = np.where(train_labels == 0)[0][0]
        # one = np.where(train_labels == 1)[0][0]
        # train_data, train_labels = train_data[[zero, one]], train_labels[[zero, one]]
        # train_loader, test_loader, val_loader = get_loaders(2, train_data, train_labels,
        #                                                     test_data, test_labels,
        #                                                     val_data, val_labels, weighted_sampler=False)

        logging.info("Data preparation completed")

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")
        start_timer()
        # Load the dataset
        # Set to train mode
        # Load the checkpoint if needed
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device_id))
            logging.info("Initializing from checkpoint: {}".format(checkpoint_path))
        else:
            logging.info("Initializing from scratch")
            model.train()

            # Set the loss
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            loss = torch.nn.NLLLoss(weight=class_weights)

            # Set the optimizer and scheduler
            optim = torch.optim.Adam(model.parameters(),
                                     lr=config.get('lr'),
                                     eps=config.get('eps'),
                                     weight_decay=config.get('weight_decay'))
            scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                        step_size=config.get('lr_scheduler_step_size'),
                                                        gamma=config.get('lr_scheduler_gamma'))

            # Train
            checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())
            #checkpoint_prefix = utils.create_output_dir('out')
            n_total_samples = 0.0
            loss_vals = []
            sample_count = []
            logging.info("Start training")
            batch_loss = None
            best_loss = -1000000
            scaler = torch.cuda.amp.GradScaler()
            for epoch in range(n_epochs):
                start = time()
                for minibatch, label in train_loader:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        minibatch = minibatch.to(device)
                        label = label.to(device)
                        # Zero the gradients
                        optim.zero_grad()
                        batch_size = label.shape[0]
                        n_total_samples += batch_size

                        # Forward pass
                        res = model(minibatch)

                        # Compute loss
                        criterion = loss(res, label)

                        # Collect for recoding and plotting
                        batch_loss = criterion.item()
                        loss_vals.append(batch_loss)
                        sample_count.append(n_total_samples)

                    # Back prop
                    scaler.scale(criterion).backward()
                    scaler.step(optim)
                    # Updates the scale for next iteration.
                    scaler.update()
                    optim.zero_grad()
                # Scheduler update
                scheduler.step()
                if epoch % n_freq_print == 0:
                    # Plot the loss function
                    loss_fig_path = checkpoint_prefix + "_loss_fig.png"
                    utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

                # Record loss on train set
                #logging.info("[Epoch-{}] loss: {:.3f}".format(epoch + 1, batch_loss))
                current_loss, current_metric = validation(model, val_loader, device, loss,
                                                          use_cuda=use_cuda)
                end = time()
                epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(n_epochs))).format(epoch + 1)
                epoch_fin = 'train: {:.6f}, val: {:.6f}, F1: {:.4f} [{}]'.format(np.mean(loss_vals), current_loss, current_metric['f1-score_macro'],
                                                               str(datetime.timedelta(seconds=(end - start))))
                logging.info(epoch_desc + epoch_fin)
                # Save checkpoint
                if best_loss > current_loss:
                    best_loss = current_loss
                    torch.save(model.state_dict(), checkpoint_prefix + "_best.pth")
                    logging.info("Best model saved")

                model.train()
            end_timer_and_print(f"Training session ({n_epochs}epochs)")

            logging.info('Training completed')
            torch.save(model.state_dict(), checkpoint_prefix + '_final.pth')

            if config.get("use_baseline"):
                model = IMUCLSBaseline(config).to(device)
            else:
                model = IMUTransformerEncoder(config).to(device)

            # Load the best model
            if checkpoint_path:
                model.load_state_dict(torch.load(checkpoint_prefix + "_best.pth", map_location=device_id))
        # Set to eval mode
        model.eval()

        metric = []
        logging.info("Start testing")
        accuracy_per_label = np.zeros(config.get("num_classes"))
        count_per_label = np.zeros(config.get("num_classes"))
        predicted = []
        ground_truth = []
        with torch.no_grad():
            for minibatch, label in test_loader:
                # Forward pass
                minibatch = minibatch.to(device)
                label = label.to(device)
                res = model(minibatch)

                # Evaluate and append
                pred_label = torch.argmax(res, dim=1)
                predicted.extend(pred_label.cpu().numpy())
                ground_truth.extend(label.cpu().numpy())
                # curr_metric = (pred_label == label).to(torch.int)
                # label_id = label[0].item()
                # accuracy_per_label[label_id] += curr_metric.item()
                # count_per_label[label_id] += 1
                # metric.append(curr_metric.item())

        metrics = get_metrics(ground_truth, predicted)
        for k, v in metrics.items():
            if "confusion" in k:
                logging.info('Fold {} {}:\n{}\n'.format(folder_idx, k.capitalize(), v))
            else:
                logging.info('Fold {} {}: {}\n'.format(folder_idx, k.capitalize(), v))

        cum_acc.append(metrics['accuracy'])
        cum_f1.append(metrics['f1-score'])
        cum_recall.append(metrics['recall'])
        cum_precision.append(metrics['precision'])
        cum_auc.append(metrics['roc_auc'])
        cum_f1_macro.append(metrics['f1-score_macro'])
        cum_recall_macro.append(metrics['recall_macro'])
        cum_precision_macro.append(metrics['precision_macro'])

    print_metrics(logging, n_classes, cum_acc, cum_recall, cum_precision, cum_auc, cum_f1, cum_recall_macro, cum_precision_macro,
                       cum_f1_macro)

    # Record overall statistics
    # confusion_mat = confusion_matrix(ground_truth, predicted, labels=list(range(config.get("num_classes"))))
    # print(confusion_mat.shape)
    # stats_msg = "\n\tAccuracy: {:.3f}".format(np.mean(metric))
    # accuracies = []
    # for i in range(len(accuracy_per_label)):
    #     print("Performance for class [{}] - accuracy {:.3f}".format(i, accuracy_per_label[i] / count_per_label[i]))
    #     accuracies.append(accuracy_per_label[i] / count_per_label[i])
    # # save dump
    # np.savez(checkpoint_path + "_test_results_dump", confusion_mat=confusion_mat, accuracies=accuracies,
    #          count_per_label=count_per_label, total_acc=np.mean(metric))
    # logging.info(stats_msg)
