import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# So that we can import from the same directory
from Dataset.ihealth import IHealth, SignalsIHealth as si

from Process.Manager import preprocess_datasets
from Process.Protocol import PatientACC
import time
import argparse


def instanciate_dataset(dir_datasets):
    datasets = []

    ihealth_fold = "/home/jsenadesouza/DA-healthy2patient/data/"

    ihealth = IHealth('IHealth', ihealth_fold, dir_datasets, freq=100, trials_per_file=10000)
    sig_ihealth = [si.acc1_dominant_wrist_X, si.acc1_dominant_wrist_Y, si.acc1_dominant_wrist_Z]
    ihealth.set_signals_use(sig_ihealth)
    datasets.append(ihealth)

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
    generate_ev = PatientACC(datasets, dir_datasets, exp_name, overlapping=overlapping,
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

    dir_datasets = '/home/jsenadesouza/DA-healthy2patient/results/dataset_preprocess/'
    dir_save_file = '/home/jsenadesouza/DA-healthy2patient/results/dataset'

    overlapping = 0
    time_wd = 5
    new_freq = 100

    if not os.path.exists(dir_datasets):
        os.makedirs(dir_datasets)
    if not os.path.exists(dir_save_file):
        os.makedirs(dir_save_file)

    datasets = instanciate_dataset(dir_datasets)

    #process_datasets(datasets)

    exp_name = "bedstatus_patientid_acc"
    create_dataset(datasets, dir_save_file, dir_datasets, exp_name,
                   overlapping, time_wd, new_freq)
    end = time.time()
    print("Time passed = {}".format(end - start), flush=True)
