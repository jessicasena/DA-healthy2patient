import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# So that we can import from the same directory
from Dataset.pain_acc_16_19 import PainAcc_16_19, SignalsPainACC as si
from Dataset.pain_acc_20_22 import PainAcc_20_22, SignalsPainACC as si2

from Process.Manager import preprocess_datasets
from Process.Protocol import painACCProtocol
import time
import argparse


def instanciate_dataset(dir_datasets):
    datasets = []
    #
    painacc_fold = "/data/datasets/ICU_Data/Sensor_Data/"
    pain_data = PainAcc_16_19('acc_pain_16_19', painacc_fold, dir_datasets, freq=100, trials_per_file=100)

    # painacc_fold = "/data/datasets/ICU_Data/354_Sensor_Data/"
    # pain_data = PainAcc_20_22('acc_pain_20_22', painacc_fold, dir_datasets, freq=100, trials_per_file=100)
    datasets.append(pain_data)

    return datasets


def process_datasets(datasets):
    # preprocessing
    print("\nDatasets preprocessing...\n", flush=True)
    preprocess_datasets(datasets)
    print("\nDone.\n", flush=True)

    return datasets


def create_dataset(datasets, dir_save_file, dir_datasets, exp_name,
                   overlapping, time_wd, new_freq):
    # Creating Loso evaluate generating
    generate_ev = painACCProtocol(datasets, dir_datasets, exp_name, overlapping=overlapping,
                               time_wd=time_wd)
    # function to save information e data
    # files = glob.glob(dir_datasets+'*.pkl')
    print("\n--Npz generating--\n", flush=True)
    generate_ev.simple_generate(dir_save_file, new_freq=new_freq)

    print("\nNpz Done.\n", flush=True)


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    dir_datasets = '/home/jsenadesouza/DA-healthy2patient/results/pain/dataset_preprocess/'
    dir_save_file = '/home/jsenadesouza/DA-healthy2patient/results/pain/dataset'

    overlapping = 0.5
    time_wd = 1800
    new_freq = 10

    if not os.path.exists(dir_datasets):
        os.makedirs(dir_datasets)
    if not os.path.exists(dir_save_file):
        os.makedirs(dir_save_file)

    datasets = instanciate_dataset(dir_datasets)

    #process_datasets(datasets)
    # sys.exit("\nDatasets preprocessing done.\n")

    exp_name = "painscore_patientid_acc_30minmargin_measurednowcol_30min_10hz_filtered"
    create_dataset(datasets, dir_save_file, dir_datasets, exp_name,
                   overlapping, time_wd, new_freq)
    end = time.time()
    print("Time passed = {}".format(end - start), flush=True)
