import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dataset.Datasets import Dataset
import numpy as np
import glob, os
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from bisect import bisect_left, bisect_right
from operator import itemgetter


def read_acc_file(file_name):
    um_acc_raw = pd.read_csv(file_name, header=10)
    return um_acc_raw.to_numpy()


def sampling_rate(data, rate_reduc):
    number_samp = data.shape[0]
    samples_slct = list(range(0,number_samp,rate_reduc))
    new_data = data[samples_slct]
    return np.array(new_data)


class SignalsPainACC():
    acc_X = 0
    acc_Y = 1
    acc_Z = 2
    patient_id = 3
    timestamp = 4
    pain_score = 5


def relabel(label):
    label = int(label)
    if label == 0:
        return "none"
    elif 1 <= label <= 4:
        return "mild"
    elif 5 <= label <= 6:
        return "moderate"
    elif 7 <= label <= 10:
        return "severe"
    else:
        return "unknown"


def diff_times_patient_pain_measurement(pain_file):
    patient_times = {}
    for row in pain_file:
        if row[1] in patient_times:
            current = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S.%f')
            patient_times[row[1]].append(current)
        else:
            current = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S.%f')
            patient_times[row[1]] = [current]
            past = current

    diff_times = {}
    for key, value in patient_times.items():
        times = []
        for i in range(1, len(value)):
            times.append(value[i] - value[i-1])
        diff_times[key] = np.mean(times)

    for key, value in diff_times.items():
        print(key, value)

def process_dvprs(value):
    try:
        score = int(value.split(' ')[0])
    except:
        if value == 'Patient Asleep':
            score = -1
        elif np.isnan(value):
            score = -1
        else:
            sys.exit('error in dvprs')

    return score

def convert_to_date(value):
    return datetime.strptime(value, '%m/%d/%Y %H:%M:%S.%f')

def add_

def add_labels(ts_series):
    labels = {}



class PainAcc_16_19(Dataset):
    def __init__(self, name, dir_dataset, dir_save, freq = 100, trials_per_file=100000, time_wd=1800, time_drop=900):
        super().__init__(name, dir_dataset, dir_save, freq, trials_per_file)
        self.reduce_rate = 10
        self.time_wd = time_wd * (self.freq/self.reduce_rate)
        self.time_drop = time_drop * (self.freq/self.reduce_rate)

    def print_info(self):
        return """
                device: IMU
                frequency: 100Hz
                positions: dominant wrist, chest and dominant side's ankle
                sensors: heart rate, temperature, acc, gyr and mag
                """

    def get_accs_files(self):
        accs = {}
        for root, dirs, files in os.walk(self.dir_dataset):
            for file in files:
                if file.endswith("RAW.csv"):
                    patient = root.split('/')[5].split("_")[1]
                    timestamp = file.split("(")[-1].split(")")[0]
                    bodypart = file.split("_")[0]
                    acc_csv = os.path.join(root, file)
                    # just get csv files from Accelerometer directories
                    if os.path.join(self.dir_dataset, f"Patient_{patient}", "Accelerometer") in root:
                        if acc_csv not in accs:
                            accs[acc_csv] = {'patient': patient, 'timestamp': timestamp, "bodypart": bodypart}
        return accs

    def get_pain_labels(self):
        file = pd.read_csv('/data/datasets/ICU_Data/EHR_Data/truncated/2020-02-26/pain_0_trimmed.csv')

        pain_map = {}
        pain_file = []
        for row in file.itertuples():
            try:
                pain_datetime = datetime.strptime(row.pain_datetime, '%Y/%m/%d %H:%M:%S')
            except:
                pain_datetime = datetime.strptime(row.pain_datetime, '%Y-%m-%d %H:%M:%S')

            pain_datetime = pain_datetime.strftime('%m/%d/%Y %H:%M:%S.%f')

            if pain_datetime in pain_map:
                if row.record_id == pain_map[pain_datetime]['patient_id']:
                    print("duplicate in dates")
            else:
                score = process_dvprs(row.pain_uf_dvprs)
                if score != -1:
                    pain_map[pain_datetime] = {'pain_uf_dvprs': score, 'patient_id': row.record_id}

        for timestamp in pain_map.keys():
            try:
                mean_pain = pain_map[timestamp]['pain_uf_dvprs']
                patient_id = pain_map[timestamp]['patient_id']
                pain_file.append([timestamp, patient_id, mean_pain])
            except:
                print(timestamp)
                print(pain_map[timestamp]['pain_uf_dvprs'])
                print(pain_map[timestamp]['patient_id'])

        pain_dict = {}
        for row in pain_file:
            pain_datetime = convert_to_date(row[0])
            pain_date = pain_datetime.date().strftime("%m/%d/%Y")
            if pain_date + "_" + str(row[1]) in pain_dict:
                pain_dict[pain_date + "_" + str(row[1])].append([pain_datetime, row[2]])

            else:
                pain_dict[pain_date + "_" + str(row[1])] = [[pain_datetime, row[2]]]

        return pain_dict

    def preprocess(self):
        accs_files = self.get_accs_files()
        pain_labels = self.get_pain_labels()
        trial_id = 1
        output_dir = self.dir_save

        for file in tqdm(accs_files.keys()):
            try:
                print(file)
                patient_id = file.split("/")[5].split("_")[1]
                init_day = datetime.strptime(file.split("/")[-2].split(" ")[1], '%m.%d.%y')
                last_day = datetime.strptime(file.split("/")[-2].split(" ")[3], '%m.%d.%y')
                current = init_day
                match = False
                pain_filtered = []
                while last_day >= current:
                    key = current.date().strftime("%m/%d/%Y") + "_" + patient_id
                    if key in pain_labels:
                        pain_filtered.extend(pain_labels[key])
                        match = True
                    current = current + timedelta(days=1)

                pain_filtered = sorted(pain_filtered, key=itemgetter(0))
                if match:

                    um_acc = read_acc_file(file)
                    # resampled the data to 10Hz
                    acc_reduced = sampling_rate(um_acc, self.reduce_rate)
                    # transform the data to a dict with the timestamp as key
                    acc_dict = list(map(convert_to_date, acc_reduced[:, 0]))

                    for pain_measurement in pain_filtered:
                        pain_datetime = pain_measurement[0]
                        idx = bisect_left(acc_dict, pain_datetime)
                        idx = idx - 1 if idx >= len(acc_dict) else idx
                        if idx > self.time_wd + self.time_drop:
                            margin = timedelta(minutes=5)
                            if abs(pain_datetime - acc_dict[idx]) <= margin:
                                # get 30 minutes before 15 minutes from the pain measurement
                                # 15 minutes are dropped because the nurse can be in the room doing some procedures
                                ts_sample = acc_reduced[int(idx - self.time_wd - self.time_drop):int(idx - self.time_drop),
                                         0]
                                # TODO: check if the timestamps in the sample are continuous
                                if ts_sample.diff().median() < timedelta(minutes=1):
                                    sample = acc_reduced[
                                             int(idx - self.time_wd - self.time_drop):int(idx - self.time_drop), 1:4]
                                    # add labels
                                    label = pain_measurement[1]
                                    self.add_info_data(str(label), patient_id, trial_id, sample, output_dir)
                                    trial_id += 1
            except:
                print("Error on file: ", file)

        self.save_data(output_dir)