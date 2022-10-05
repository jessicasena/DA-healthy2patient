import glob
import sys
import os
import fnmatch
import pickle
import cupy as np
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# So that we can import from the same directory
import time
from loguru import logger
from dataset import PainDataset
from dask_cuda import LocalCUDACluster
from dask.distributed import Client


def process_studies(dir_save, trials_per_file, time_wd, time_drop, final_freq, logger):

    data_name = []
    # Intelligent ICU study
    # i_icu = PainDataset(dataset_name="intelligent_icu", dir_dataset="/data/datasets/ICU_Data/Sensor_Data/",
    #             dir_save=dir_save, trials_per_file=trials_per_file, time_wd=time_wd, time_drop=time_drop,
    #             final_freq=final_freq, logger=logger)
    # i_icu.preprocess()
    # data_name.append("intelligent_icu")

    # PAIN study
    pain = PainDataset(dataset_name="pain", dir_dataset="/home/jsenadesouza/DA-healthy2patient/354_Sensor_data/",
                dir_save=dir_save, trials_per_file=trials_per_file, time_wd=time_wd, time_drop=time_drop,
                final_freq=final_freq, logger=logger)
    pain.preprocess()
    data_name.append("pain")

    # ADAPT study
    adapt = PainDataset(dataset_name="adapt", dir_dataset="/home/jsenadesouza/DA-healthy2patient/1013_Sensor_Data/",
                dir_save=dir_save, trials_per_file=trials_per_file, time_wd=time_wd, time_drop=time_drop,
                final_freq=final_freq, logger=logger)
    adapt.preprocess()
    data_name.append("adapt")

    return data_name


def data_generator(X, y, files, data_name, dir_input_file):
    print("\nAdding samples from {}".format(data_name), flush=True)
    count_samples = 0
    count_patients = []
    for id_, fl in enumerate(files):
        pkl = os.path.join(dir_input_file, data_name + '_' + str(id_) + '.pkl')
        with open(pkl, 'rb') as handle:
            data = pickle.load(handle)
            fl = [i for i in data]
            for file in fl:
                label_vector = file.split("_")[:17]
                subject_id = file.split("_")[17].split("s")[1]
                label_vector.append(subject_id)
                # TODO - check if this is correct
                if data_name == "acc_outcomes_PAIN" or data_name == "acc_outcomes_ADAPT":
                    trial = np.array(data[file])[:, 1:]
                else:
                    trial = np.squeeze(np.array(data[file]))

                count_samples += 1
                if subject_id not in count_patients:
                    count_patients.append(subject_id)

                X.append(np.array([trial]))
                y.append(label_vector)
    print(f"{count_samples} samples\n")
    print(f"{len(count_patients)} patients")
    return X, y


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
        X, y = data_generator(X, y, files_s[dataset_name], dataset_name, dir_pkl)

    X = np.array(X, dtype=float)
    y = np.array(y)

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

    y_col_names = ['heart_rate', 'heart_rate_class', 'temp', 'temp_class', 'lenght_of_stay', 'is_dead', 'pain_score',
                   'pain_score_class',
                   'sofa_score', 'sofa_score_class', 'map', 'map_class', 'braden_score', 'braden_score_class', 'spo2',
                   'spo2_class', 'cam', 'patient_id']
    np.savez_compressed(os.path.join(dir_save_file, name_file),
                        X=X,
                        y=y,
                        y_col_names=y_col_names)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Create a Dask Cluster with one worker per GPU
    cluster = LocalCUDACluster()
    client = Client(cluster)
    start = time.time()
    exp_name = "IntelligentICU_PAIN_ADAPT_15min"
    dir_save_datasets = '/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset_preprocess_15min/'
    dir_save_file = '/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset'
    time_wd = 900
    time_drop = 900
    new_freq = 10
    trials_per_file = 100

    logger.remove(0)
    logger.add('/home/jsenadesouza/DA-healthy2patient/results/outcomes/' + exp_name + ".log", enqueue=True,
               format="{time} | {level} | {message}", colorize=True, mode="w")
    logger.add(sys.stderr, enqueue=True, format="{time} | {level} | {message}", colorize=True)

    if not os.path.exists(dir_save_datasets):
        os.makedirs(dir_save_datasets)
    if not os.path.exists(dir_save_file):
        os.makedirs(dir_save_file)

    list_datasets = process_studies(dir_save_datasets, trials_per_file, time_wd, time_drop, new_freq, logger)
    generate_dataset(dir_save_datasets, list_datasets)

    end = time.time()
    print("Time passed = {}".format(end - start), flush=True)
