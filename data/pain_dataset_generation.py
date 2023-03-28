import glob
import shutil
import sys
import os
import fnmatch
import pickle
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# So that we can import from the same directory
import time
from loguru import logger
from dataset import PainDataset
from dask_cuda import LocalCUDACluster
from dask.distributed import Client


def process_studies(dir_save, trials_per_file, time_wd, time_drop, final_freq, logger, margin, process=True):
    # instantiate a object for each dataset and process the data
    data_name = []
    #Intelligent ICU study
    i_icu = PainDataset(dataset_name="intelligent_icu", dir_dataset="/data/datasets/ICU_Data/Sensor_Data/",
                dir_save=dir_save, trials_per_file=trials_per_file, time_wd=time_wd, time_drop=time_drop,
                final_freq=final_freq, logger=logger)
    if process:
      i_icu.preprocess(margin)
    data_name.append("intelligent_icu")

    # PAIN study
    pain = PainDataset(dataset_name="pain", dir_dataset="/home/jsenadesouza/DA-healthy2patient/354_Sensor_data/",
                dir_save=dir_save, trials_per_file=trials_per_file, time_wd=time_wd, time_drop=time_drop,
                final_freq=final_freq, logger=logger)
    if process:
        pain.preprocess(margin)
    data_name.append("pain")

    # ADAPT study
    adapt = PainDataset(dataset_name="adapt", dir_dataset="/home/jsenadesouza/DA-healthy2patient/1013_Sensor_Data/",
                dir_save=dir_save, trials_per_file=trials_per_file, time_wd=time_wd, time_drop=time_drop,
                final_freq=final_freq, logger=logger)
    if process:
        adapt.preprocess(margin)
    data_name.append("adapt")

    return data_name


def data_generator(X, y, files, data_name):
    print("\nAdding samples from {}".format(data_name), flush=True)
    count_samples = 0
    count_patients = []
    for pkl in files:
        # open pickle file containing the trials
        # (1 trial = 1 sample of 15 minutes of accelerometer data extracted before a pain measurement)
        with open(pkl, 'rb') as handle:
            data = pickle.load(handle)
            fl = [i for i in data]
            for trial in fl:
                label_vector = trial.split("_")[:-2]
                subject_id = trial.split("_")[-2].split("s")[1]
                label_vector.append(subject_id)
                # get the accel data
                trial_accel = np.squeeze(data[trial].get())

                count_samples += 1
                if subject_id not in count_patients:
                    count_patients.append(subject_id)
                # save accel data and labels
                X.append(trial_accel)
                y.append(label_vector)
    print(f"{count_samples} samples\n")
    print(f"{len(count_patients)} patients")
    return X, y


def plot_accel(sample, title, path, idx):

    fig = plt.figure(idx)
    plt.plot(sample[:, 0], label="x")
    plt.plot(sample[:, 1], label="y")
    plt.plot(sample[:, 2], label="z")
    plt.title(title)
    plt.legend()

    plt.savefig(os.path.join(path,f"{title}_{time.time()}.png"))
    plt.close(fig)


def generate_dataset(dir_pkl, list_datasets):
    name_file = 't{}_{}'.format(time_wd, exp_name)
    X, y = [], []

    files_s = {}
    for dataset_name in list_datasets:
        files_s[dataset_name] = []
        files = glob.glob(os.path.join(dir_pkl, '*.pkl'))
        for pkl in files:
            if fnmatch.fnmatch(os.path.split(pkl)[-1], dataset_name + '_*.pkl'):
                files_s[dataset_name].append(pkl)
    print("Done.", flush=True)

    for dataset_name in list_datasets:
        X, y = data_generator(X, y, files_s[dataset_name], dataset_name)

    X = np.array(X)
    y = np.array(y)

    # remove samples that the entire content is the same value
    new_X = []
    new_y = []
    count_bad_data = 0
    for xx, yy in zip(X, y):
        if np.min(xx) == np.max(xx):
            count_bad_data += 1
        else:
            new_X.append(xx)
            new_y.append(yy)

    print(f"{count_bad_data} bad samples")
    X = new_X
    y = new_y

    # # remove samples with no movement
    # new_X = []
    # new_y = []
    # var_threshold = 0.01
    # count_no_movement = 0
    # for sample, label in zip(X, y):
    #     sample = np.squeeze(sample)
    #     var = np.mean([np.var(sample[:, 0]), np.var(sample[:, 1]), np.var(sample[:, 2])])
    #     if var > var_threshold:
    #         new_X.append(sample)
    #         new_y.append(label)
    #     else:
    #         count_no_movement += 1
    #         path = os.path.join("/home/jsenadesouza/DA-healthy2patient/results/outcomes/", "no_movement")
    #         plot_accel(sample, label[7], path, count_no_movement)
    # print("{} samples removed because of no movement".format(count_no_movement))
    # X = new_X
    # y = new_y

    # store the name of the labels columns
    y_col_names = ['heart_rate', 'heart_rate_class', 'temp', 'temp_class', 'lenght_of_stay', 'is_dead', 'pain_score',
                   'pain_score_class',  'pain_score_prev', 'pain_score_prev_class',
                   'sofa_score', 'sofa_score_class', 'map', 'map_class', 'braden_score', 'braden_score_class', 'spo2',
                   'spo2_class', 'cam', 'timestamp', 'patient_id']
    # save to a npz fila the accel data and labels and column names
    np.savez_compressed(os.path.join(dir_save_file, name_file),
                        X=X,
                        y=y,
                        y_col_names=y_col_names)

    n_samples = len(X)
    n_patients = len(np.unique(np.asarray(y)[:, -1])) if len(y) > 0 else 0

    return n_samples, n_patients


if __name__ == "__main__":
    start = time.time()
    process = True
    if process:
        multiprocessing.freeze_support()
        # Create a Dask Cluster to use multiple GPUs
        cluster = LocalCUDACluster()
        client = Client(cluster)
    margin_samples = []
    count = 0
    for margin in [10, 20, 30]:
        exp_name = "INTELLIGENT_ADAPT_PAIN_15wd_15drop_painprev_timestamp_bugfixed_margin{}".format(margin)
        dir_save_datasets = '/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset_preprocess_15min/'
        dir_save_file = '/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset'
        time_wd = 900
        time_drop = 9000
        new_freq = 10
        trials_per_file = 100
        if count == 0:
            logger.remove(0)
        count += 1
        logger.add('/home/jsenadesouza/DA-healthy2patient/results/outcomes/' + exp_name + ".log", enqueue=True,
                   format="{time} | {level} | {message}", colorize=True, mode="w")
        logger.add(sys.stdout, format="[{time}] [<level>{level}</level>] {message}", colorize=True)

        if os.path.exists(dir_save_datasets):
            shutil.rmtree(dir_save_datasets)
        os.makedirs(dir_save_datasets)
        if not os.path.exists(dir_save_file):
            os.makedirs(dir_save_file)

        list_datasets = process_studies(dir_save_datasets, trials_per_file, time_wd, time_drop, new_freq, logger, margin, process=process)
        n_samples, n_patients = generate_dataset(dir_save_datasets, list_datasets)
        margin_samples.append([margin, n_samples, n_patients])

        end = time.time()
        print("Time passed = {}".format(end - start), flush=True)

    print("Margin experiment finished")
    print(margin_samples)
