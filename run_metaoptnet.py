# -*- coding: utf-8 -*-
import os
import argparse
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MetaOptNet.classification_heads import ClassificationHead
from utils.models import ProtoMetaSenseModel

from MetaOptNet.utils_metaopt import set_gpu, Timer, count_accuracy, check_dir, log
from utils.sensordata import SensorDataset, FewShotDataloader
from utils.utils import get_metrics
import time
import datetime
import scipy.stats as st

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if torch.cuda.is_available():
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    network = ProtoMetaSenseModel()
    if torch.cuda.is_available():
        network = network.cuda()
    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet')
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge')
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2')
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS')
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
    if torch.cuda.is_available():
        cls_head = cls_head.cuda()
    return (network, cls_head)


def get_dataset(args, phase, fold=0):
    if phase == 'train':
        dataset_train = SensorDataset(args.file_path, phase='train')
        dataset_val = SensorDataset(args.file_path, phase='val')
        data_loader = FewShotDataloader

        return (dataset_train, dataset_val, data_loader)
    else:
        dataset_test = SensorDataset(args.file_path, num_shots=args.shot, fold=fold, phase='test')
        data_loader = FewShotDataloader

        return (dataset_test, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='path to the dataset')
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--test-query', type=int, default=15,
                       help='number of query examples per training class')
    parser.add_argument('--test-episode', type=int, default=1000,
                        help='number of episodes to test')
    parser.add_argument('--way', type=int, default=3,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--results_path', default='./experiments/exp_1')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')

    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    opt = parser.parse_args()

    start = time.time()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    (dataset_train, dataset_val, data_loader) = get_dataset(opt, 'train')

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.way * opt.train_query, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=0,
        epoch_size=opt.episodes_per_batch * 1000, # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu("0")
    folder_name = os.path.split(opt.file_path)[-1].split('.npz')[0]
    results_path = os.path.join(opt.results_path, folder_name, f'_{opt.shot}shot')  # save results in this folder

    os.makedirs(results_path, exist_ok=True)
    
    log_file_path = os.path.join(results_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    
    #lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.num_epoch // 6, gamma=0.5)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()
        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() if torch.cuda.is_available() else x for x in batch]

            train_n_support = opt.way * opt.shot
            train_n_query = opt.way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-2:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
            
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-2:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
            
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            
            acc = count_accuracy(logit_query.reshape(-1, opt.way), labels_query.reshape(-1))
            
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        
        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() if torch.cuda.is_available() else x for x in batch]

            test_n_support = opt.way * opt.shot
            test_n_query = opt.way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-2:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-2:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)

            loss = x_entropy(logit_query.reshape(-1, opt.way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(results_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(results_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(results_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))


# ---------------------------- test -------------------------------------------------
    results = {}
    for i in range(5):

        (dataset_test, data_loader) = get_dataset(opt, phase='test', fold=i)

        dloader_test = data_loader(
            dataset=dataset_test,
            nKnovel=opt.way,
            nKbase=0,
            nExemplars=opt.shot,  # num training examples per novel category
            nTestNovel=opt.test_query * opt.way,  # num test examples for all the novel categories
            nTestBase=0,  # num test examples for all the base categories
            batch_size=1,
            num_workers=0,
            epoch_size=opt.test_episode,  # num of batches per epoch
        )

        log_file_path = os.path.join(results_path, "test_log.txt")
        log(log_file_path, str(vars(opt)))

        # Define the models
        (embedding_net, cls_head) = get_model(opt)

        # Load saved model checkpoints
        saved_models = torch.load(os.path.join(results_path, 'best_model.pth'))
        embedding_net.load_state_dict(saved_models['embedding'])
        embedding_net.eval()
        cls_head.load_state_dict(saved_models['head'])
        cls_head.eval()

        # Evaluate on test set
        test_accuracies = []
        fold_results = {}
        for i, batch in enumerate(tqdm(dloader_test()), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() if torch.cuda.is_available() else x for x
                                                                            in batch]

            n_support = opt.way * opt.shot
            n_query = opt.way * opt.test_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-2:])))
            emb_support = emb_support.reshape(1, n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-2:])))
            emb_query = emb_query.reshape(1, n_query, -1)

            if opt.head == 'SVM':
                logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, maxIter=3)
            else:
                logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)

            if torch.cuda.is_available():
                labels_query = labels_query.reshape(-1).cpu()
                logits = logits.reshape(-1, opt.way).cpu().detach()
            metrics = get_metrics(np.argmax(logits, axis=1), labels_query)
            for key, value in metrics.items():
                if key in fold_results:
                    fold_results[key].append(value)
                else:
                    fold_results[key] = [value]

        for key, value in fold_results.items():
            if key != 'confusion_matrix':
                if key in results:
                    results[key].append(np.mean(value))
                else:
                    results[key] = [np.mean(value)]

        # Save results
        end = time.time()
        duration = datetime.timedelta(seconds=end - start)
        outfile = open(os.path.join(results_path, f'results_metaoptnet_5fold_{opt.shot}shot.txt'), 'w', buffering=1)
        outfile.write('Time: {}\n'.format(str(duration)))

        for met, values in results.items():
            if met != 'confusion_matrix':
                ic_acc = st.t.interval(0.9, len(values) - 1, loc=np.mean(values), scale=st.sem(values))

                outfile.write(f'{met} per fold\n')
                outfile.write('\n'.join(str(item) for item in values))
                outfile.write('\n______________________________________________________\n')
                outfile.write('Mean {} [{:.4} ± {:.4}] IC [{:.4}, {:.4}]\n\n'.format(met,
                                                                                     np.mean(values) * 100, (
                                                                                                 np.mean(values) -
                                                                                                 ic_acc[0]) * 100,
                                                                                     ic_acc[0] * 100, ic_acc[1] * 100))

