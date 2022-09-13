import multiprocessing
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


def read_acc_file(file_name, logger):
    um_acc_raw = pd.read_csv(file_name, header=10)
    if "Timestamp" not in um_acc_raw.columns[0]:
        logger.error("File {} has no timestamp column".format(file_name))
        return None
    um_acc_raw = um_acc_raw.to_numpy()

    return um_acc_raw


def sampling_rate(data, rate_reduc):
    number_samp = data.shape[0]
    samples_slct = list(range(0,number_samp,int(rate_reduc)))
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
    def __init__(self, name, dir_dataset, dir_save, logger, final_freq = 10, trials_per_file=100000, time_wd=1800, time_drop=900):
        super().__init__(name, dir_dataset, dir_save, trials_per_file=trials_per_file)
        self.time_wd = time_wd
        self.final_freq = final_freq
        self.time_drop = time_drop
        self.logger = logger
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
        patients = {}
        for root, dirs, files in os.walk(self.dir_dataset):
            for file in files:
                if file.endswith("RAW.csv"):
                    patient = root.split('/')[5].split("_")[1]
                    timestamp = file.split("(")[-1].split(")")[0]
                    bodypart = file.split("_")[0]
                    acc_csv = os.path.join(root, file)
                    if patient not in patients:
                        patients[patient] = False
                    # just get csv files from Accelerometer directories
                    if os.path.join(self.dir_dataset, f"Patient_{patient}", "Accelerometer") in root:
                        patients[patient] = True
                        if acc_csv not in accs:
                            accs.append(acc_csv)

        for pat, acc_flag in patients.items():
            if not acc_flag:
                self.logger.error("Patient {}, message: not accelerometer in directory", pat)

        return accs

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
        accs_files = self.get_accs_files()
        trial_id = 1
        output_dir = self.dir_save

        for file in tqdm(accs_files):
            samples_extracted = False
            try:
                placement = file.split("/")[-1].split("_")[0].lower()
                if placement != 'wrist' and placement != 'arm' and placement != 'emg':
                    self.logger.error("File {}, message: not wrist or arm", file)
                else:
                    self.logger.info("File {}, message: processing", file)
                    patient_id = file.split("/")[5].split("_")[1]
                    init_day = datetime.strptime(file.split("/")[-2].split(" ")[1], '%m.%d.%y')
                    last_day = datetime.strptime(file.split("/")[-2].split(" ")[3], '%m.%d.%y')
                    last_day = last_day + timedelta(hours=23, minutes=59)
                    outcomes_filtered_df, pain_filtered = self.get_labels(patient_id, init_day, last_day)
                    if len(outcomes_filtered_df) > 0 and len(pain_filtered) > 0:
                        file_acc = read_acc_file(file, self.logger)
                        if file_acc is not None:
                            # converting string to datetime
                            pool = mp.Pool()
                            file_acc[:, 0] = list(pool.map(convert_to_date, file_acc[:, 0]))

                            # resampled the data

                            df = pd.DataFrame(file_acc[:, 0])
                            self.freq = 1/df.diff().median()[0].total_seconds()
                            reduce_rate = self.freq / self.final_freq
                            file_acc = sampling_rate(file_acc, reduce_rate)
                            acc_ts_list = file_acc[:, 0]

                            time_wd = self.time_wd * (self.freq / reduce_rate)
                            time_drop = self.time_drop * (self.freq / reduce_rate)
                            self.logger.info("File {}, message: Frequency: {}, Reduce rate: {}, Time window: {}", file,
                                             self.freq, reduce_rate, time_wd)

                            for pain_datetime in pain_filtered:

                                idx = bisect_left(acc_ts_list, pain_datetime)
                                idx = idx - 1 if idx >= len(acc_ts_list) else idx
                                if idx > time_wd + time_drop:
                                    margin = timedelta(minutes=5)
                                    if abs(pain_datetime - acc_ts_list[idx]) <= margin:
                                        # get 30 minutes before 15 minutes from the pain measurement
                                        # 15 minutes are dropped because the nurse can be in the room doing some procedures
                                        start_idx = int(idx - time_wd - time_drop)
                                        end_idx = int(idx - time_drop)

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
            except Exception as e:
                self.logger.critical("File {}, message: {}", file, e)
            if not samples_extracted:
                self.logger.error("File {}, message: no sample extracted", file)
            else:
                self.logger.success("File {}, message: sample extracted", file)

        self.save_data(output_dir)

if __name__ == "__main__":
    outcomesacc_fold = "/data/datasets/ICU_Data/Sensor_Data/"
    dir_datasets = '/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset_preprocess/'
    from loguru import logger

    logger.add('/home/jsenadesouza/DA-healthy2patient/results/outcomes/' + "test" + ".log", enqueue=True,
               format="{time} | {level} | {message}", colorize=True)
    outcomes_data = Outcomes_16_19('acc_outcomes_16_19', outcomesacc_fold, dir_datasets, logger, final_freq=10, trials_per_file=100)
    outcomes_data.preprocess()