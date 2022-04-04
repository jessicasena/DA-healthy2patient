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
import fnmatch


def read_acc_file(file_name):
    um_acc_raw = pd.read_csv(file_name, header=2, delimiter='\t')
    um_acc_raw = um_acc_raw.to_numpy()
    timestamp = list(map(lambda x: datetime.fromtimestamp(x / 1000), um_acc_raw[:, 0]))
    um_acc = np.hstack([np.expand_dims(timestamp, axis=1), um_acc_raw[:, 1:]])
    return um_acc


def sampling_rate(data, rate_reduc):
    number_samp = data.shape[0]
    samples_slct = list(range(0,number_samp,rate_reduc))
    new_data = data[samples_slct]
    return np.array(new_data)


class SignalsPainACC():
    timestamp = 0
    acc_X = 1
    acc_Y = 2
    acc_Z = 3


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


class PainAcc_20_22(Dataset):
    def __init__(self, name, dir_dataset, dir_save, freq = 100, trials_per_file=100000, time_wd=1800, time_drop=900):
        super().__init__(name, dir_dataset, dir_save, freq, trials_per_file)
        self.reduce_rate = 10
        self.time_wd = time_wd * (self.freq/self.reduce_rate)
        self.time_drop = time_drop * (self.freq/self.reduce_rate)
        self.patient_map = self.set_patient_map()

    def print_info(self):
        return """
                device: IMU
                frequency: 100Hz
                positions: dominant wrist, chest and dominant side's ankle
                sensors: heart rate, temperature, acc, gyr and mag
                """

    def set_patient_map(self):
        # create a map between the subject_deiden_id and the patient id
        patient_map = {}
        patient_enrollment = pd.read_excel('/data/daily_data/enrollment_log.xlsx')

        for row in patient_enrollment.itertuples():
            patient_map[row.subject_deiden_id] = row._6

        return patient_map

    def get_accs_files(self):
        accs = []
        for root, dirs, files in os.walk(self.dir_dataset):
            for file in files:
                if file.endswith("SD.csv"):
                    acc_csv = os.path.join(root, file)
                    # just get csv files from Accelerometer directories
                    if fnmatch.fnmatch(root, f'{self.dir_dataset}*/Accel/*'):
                        if acc_csv not in accs:
                            accs.append(acc_csv)
        return accs

    def get_pain_labels(self):
        pain_map = {}
        pain_file = []

        # get all files
        files = glob.glob('/data/daily_data/*/pain*.csv',
                          recursive=True)
        for file in files:
            try:
                df = pd.read_csv(file)
                file_filtered = df[df["measurement_name"] == "pain_uf_dvprs"]
            except:
                col_names = ['pain_datetime', 'measurement_name', 'measurement_value', 'patient_deiden_id',
                             'encounter_deiden_id']
                df = pd.read_csv(file, names=col_names, header=None)
                file_filtered = df[df["measurement_name"] == "pain_uf_dvprs"]

            for row in file_filtered.itertuples():
                try:
                    pain_datetime = datetime.strptime(row.pain_datetime, '%Y/%m/%d %H:%M:%S')
                except:
                    pain_datetime = datetime.strptime(row.pain_datetime, '%Y-%m-%d %H:%M:%S')

                pain_datetime = pain_datetime.strftime('%m/%d/%Y %H:%M:%S.%f')

                if pain_datetime in pain_map:
                    if row.patient_deiden_id == pain_map[pain_datetime]['patient_id']:
                        print("duplicate in dates")
                else:
                    score = process_dvprs(row.measurement_value)
                    if score != -1:
                        try:
                            pain_map[pain_datetime] = {'pain_uf_dvprs': score,
                                                       'patient_id': self.patient_map[row.patient_deiden_id]}
                        except KeyError as e:
                            # print(e)
                            pass

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

        for file in tqdm(accs_files):
            #try:

            patient_id = file.split("P")[1].split("/")[0]
            file_timestamp = datetime.strptime(file.split("/")[-2].split("_")[0], '%Y-%m-%d')

            match = False
            pain_filtered = []
            key = file_timestamp.date().strftime("%m/%d/%Y") + "_" + str(int(patient_id))
            if key in pain_labels:
                pain_filtered.extend(pain_labels[key])
                match = True

            pain_filtered = sorted(pain_filtered, key=itemgetter(0))
            if match:
                print(file)

                um_acc = read_acc_file(file)
                # resampled the data to 10Hz
                acc_reduced = sampling_rate(um_acc, self.reduce_rate)
                # transform the data to a dict with the timestamp as key
                acc_dict = acc_reduced[:, 0]

                for pain_measurement in pain_filtered:
                    pain_datetime = pain_measurement[0]
                    idx = bisect_left(acc_dict, pain_datetime)
                    idx = idx - 1 if idx >= len(acc_dict) else idx
                    if idx > self.time_wd + self.time_drop:
                        margin = timedelta(minutes=5)
                        if abs(pain_datetime - acc_dict[idx]) <= margin:
                            # get 30 minutes before 15 minutes from the pain measurement
                            # 15 minutes are dropped because the nurse can be in the room doing some procedures
                            sample = acc_reduced[int(idx - self.time_wd - self.time_drop):int(idx - self.time_drop), 1:4]
                            label = pain_measurement[1]
                            self.add_info_data(str(label), patient_id, trial_id, sample, output_dir)
                            trial_id += 1
            else:
                print(f'NOT MATCHED: {file}')
            # except:
            #     print("Error on file: ", file)

        self.save_data(output_dir)