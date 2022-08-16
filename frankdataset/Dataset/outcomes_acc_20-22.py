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
import multiprocessing as mp
import fnmatch

def read_acc_file(file_name):
    um_acc_raw = pd.read_csv(file_name, header=10)
    um_acc_raw = um_acc_raw.to_numpy()
    # resampled the data to 10Hz
    um_acc_raw = sampling_rate(um_acc_raw, 10)
    return um_acc_raw


def sampling_rate(data, rate_reduc):
    number_samp = data.shape[0]
    samples_slct = list(range(0,number_samp,rate_reduc))
    new_data = data[samples_slct]
    return np.array(new_data)

def convert_to_date(value):
    return datetime.strptime(value, '%m/%d/%Y %H:%M:%S.%f')


def process_labels(df, start_ts, end_ts):
    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
    if len(df) == 0:
        return []
    else:
        heart_rate = df["heart_rate"][df["heart_rate"] != -1]
        heart_rate = np.mean(heart_rate) if len(heart_rate) > 0 else -1
        if heart_rate == -1:
            heart_rate_class = -1
        elif 0 <= heart_rate < 60:
            heart_rate_class = "low"
        elif 60 <= heart_rate <= 100:
            heart_rate_class = "normal"
        else:
            heart_rate_class = "high"
        temp = df["temp_farenheit"][df["temp_farenheit"] != -1]
        temp = np.mean(temp) if len(temp) > 0 else -1
        if temp == -1:
            temp_class = -1
        elif 0 <= temp < 97:
            temp_class = "low"
        elif 97 <= temp <= 99:
            temp_class = "normal"
        else:
            temp_class = "high"
        lenght_of_stay = df["lenght_stay"][df["lenght_stay"] != -1]
        lenght_of_stay = np.mean(lenght_of_stay) if len(lenght_of_stay) > 0 else -1
        is_dead = df["is_dead"][df["is_dead"] != -1]
        is_dead = np.argmax(np.bincount(is_dead)) if len(is_dead) > 0 else -1
        pain_score = df["pain_score"][df["pain_score"] != -1]
        pain_score = np.mean(pain_score) if len(pain_score) > 0 else -1
        if pain_score == -1:
            pain_score_class = -1
        elif 0 <= pain_score < 5:
            pain_score_class = "mild"
        else:
            pain_score_class = "severe"
        sofa_score = df["sofa_score"][df["sofa_score"] != -1]
        sofa_score = np.mean(sofa_score) if len(sofa_score) > 0 else -1
        if sofa_score == -1:
            sofa_score_class = -1
        elif 0 <= sofa_score <= 9:
            sofa_score_class = "low"
        elif 9 < sofa_score <= 14:
            sofa_score_class = "moderate"
        else:
            sofa_score_class = "high"
        map = df["blood_pressure"][df["blood_pressure"] != '-1']
        if len(map) > 0:
            map = np.mean(map)
            if map < 70:
                map_class = 'low'
            elif 60 <= map <= 100:
                map_class = 'normal'
            else:
                map_class = 'high'
        else:
            map_class = -1

        braden_score = df["braden_score"][df["braden_score"] != -1]
        braden_score = np.mean(braden_score) if len(braden_score) > 0 else -1
        if braden_score == -1:
            braden_score_class = -1
        elif 0 <= braden_score <= 14:
            braden_score_class = "high"
        else:
            braden_score_class = "mild"
        spo2 = df["spo2"][df["spo2"] != -1]
        spo2 = np.mean(spo2) if len(spo2) > 0 else -1
        if spo2 == -1:
            spo2_class = -1
        elif 0 <= spo2 >= 95:
            spo2_class = "normal"
        else:
            spo2_class = "low"
        cam = df["cam"][df["cam"] != -1]
        cam = np.argmax(np.bincount(cam)) if len(cam) > 0 else -1
        return np.nan_to_num(
            [heart_rate, heart_rate_class, temp, temp_class, lenght_of_stay, is_dead, pain_score, pain_score_class,
             sofa_score, sofa_score_class, map, map_class, braden_score, braden_score_class, spo2, spo2_class, cam],
            nan=-1)


class Outcomes_16_19(Dataset):
    def __init__(self, name, dir_dataset, dir_save, freq = 100, trials_per_file=100000, time_wd=1800, time_drop=900):
        super().__init__(name, dir_dataset, dir_save, freq, trials_per_file)
        self.reduce_rate = 10
        self.time_wd = time_wd * (self.freq/self.reduce_rate)
        self.time_drop = time_drop * (self.freq/self.reduce_rate)
        self.outcomes_file = self.read_labels_file()

    def print_info(self):
        return """
                device: IMU
                frequency: 100Hz
                positions: dominant wrist, chest and dominant side's ankle
                sensors: heart rate, temperature, acc, gyr and mag
                """

    def get_accs_files(self):
        accs = []
        no_acc = 0
        for root, dirs, files in os.walk(self.dir_dataset):
            for file in files:
                if file.endswith("SD.csv"):
                    acc_csv = os.path.join(root, file)
                    # just get csv files from Accelerometer directories
                    if fnmatch.fnmatch(root, f'{self.dir_dataset}*/Accel/*'):
                        if acc_csv not in accs:
                            accs.append(acc_csv)
                    else:
                        no_acc += 1

        return accs, no_acc

    def read_labels_file(self):
        df = pd.read_csv(
            '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/2016-20119_clinical_data_outcomes.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def get_labels(self, patient_id, init_day, last_day):
        df = self.outcomes_file
        df = df[(df['patient_id'] == int(patient_id)) & (df['timestamp'] >= init_day) & (df['timestamp'] <= last_day)]
        df = df.sort_values(by=['timestamp'])
        pain_measurements = df[(df['pain_score'] != -1)]
        pain_measurements = pain_measurements['timestamp'].dt.to_pydatetime()
        return df, pain_measurements

    def preprocess(self):
        accs_files, no_acc = self.get_accs_files()
        trial_id = 1
        output_dir = self.dir_save

        for file in tqdm(accs_files.keys()):
            samples_extracted = False
            try:
                if 'wrist' != file and 'arm' != file:
                    print("\nDiscarding: ", file)
                else:
                    print(f'\nKeeping: {file}')
                    file_acc = read_acc_file(file)
                    init_day = 0
                    last_day = 1
                    #last_day = last_day + timedelta(hours=23, minutes=59)

                    patient_id = file.split("/")[5].split("_")[1]

                    outcomes_filtered_df, pain_filtered = self.get_labels(patient_id, init_day, last_day)
                    if len(outcomes_filtered_df) > 0 and len(pain_filtered) > 0:


                        # converting string to datetime
                        pool = mp.Pool()
                        file_acc[:, 0] = list(pool.map(convert_to_date, file_acc[:, 0]))
                        acc_ts_list = file_acc[:, 0]

                        for pain_datetime in pain_filtered:

                            idx = bisect_left(acc_ts_list, pain_datetime)
                            idx = idx - 1 if idx >= len(acc_ts_list) else idx
                            if idx > self.time_wd + self.time_drop:
                                margin = timedelta(minutes=5)
                                if abs(pain_datetime - acc_ts_list[idx]) <= margin:
                                    # get 30 minutes before 15 minutes from the pain measurement
                                    # 15 minutes are dropped because the nurse can be in the room doing some procedures
                                    start_idx = int(idx - self.time_wd - self.time_drop)
                                    end_idx = int(idx - self.time_drop)

                                    ts_sample = file_acc[start_idx:end_idx, 0]
                                    # check if the timestamps in the sample are continuous
                                    if np.mean(np.diff(ts_sample)) < timedelta(minutes=1):
                                        start_ts = ts_sample[0]
                                        end_ts = ts_sample[-1]
                                        sample = file_acc[start_idx:end_idx, 1:4]
                                        # add labels
                                        label = process_labels(outcomes_filtered_df, start_ts, end_ts)
                                        if len(label) > 0:
                                            label = "_".join(label.astype(str))
                                            self.add_info_data(label, patient_id, trial_id, sample, output_dir)
                                            trial_id += 1
                                            samples_extracted = True
            except:
                print("Error on file: ", file)
            if not samples_extracted:
                print(f'No samples extracted for {file}')

        self.save_data(output_dir)