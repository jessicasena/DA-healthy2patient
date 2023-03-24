import sys
from sklearn.utils import class_weight

from utils.data import SensorDataset

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
from utils.util import get_metrics, print_metrics, start_timer, end_timer_and_print
from models.util import load_data_mag, set_logger, split_data, get_loaders
from models.IMU_Transformer.trans_models import bigbird_pegasus
import torchtest as tt


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

    data_path = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/t900_INTELLIGENT_PAIN_ADAPT_15min.npz"
    checkpoint_path = None
    #checkpoint_path = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/IMUTransformers/out/run_10_10_22_22_20_final.pth"
    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    #device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
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
    X, y, y_target = load_data_mag(data_path, clin_variable_target="pain_score_class")
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
        # train_loader, test_loader, val_loader = get_loaders(config.get("batch_size"), train_data, train_labels, test_data, test_labels,
        #                                                     val_data, val_labels, weighted_sampler=True)

        # zero = np.where(train_labels == 0)[0][0]
        # one = np.where(train_labels == 1)[0][0]
        # train_data, train_labels = train_data[[zero, one]], train_labels[[zero, one]]
        train_set = bigbird_pegasus.PegasusDataset(train_data, train_labels)
        test_set = bigbird_pegasus.PegasusDataset(test_data, test_labels)
        val_set = bigbird_pegasus.PegasusDataset(val_data, val_labels)

        logging.info("Data preparation completed")

        model_name = "hf-internal-testing/tiny-random-bigbird_pegasus"

        #train_dataset, _, _, tokenizer = bigbird_pegasus.prepare_data(model_name, train_data, train_labels)
        trainer = bigbird_pegasus.prepare_fine_tuning(model_name, train_set)
        trainer.train()
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
