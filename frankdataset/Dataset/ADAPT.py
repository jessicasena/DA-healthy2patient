import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dataset.Datasets import Dataset
import numpy as np
import glob, os
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta
import datetime as dt
from tqdm import tqdm
from bisect import bisect_left, bisect_right
from operator import itemgetter
import multiprocessing as mp
import fnmatch
import dateutil
from zoneinfo import ZoneInfo
import multiprocessing

def parse_date(x):
    return dateutil.parser.parse(x)

def read_acc_file(file_name, outfile):
    um_acc_raw = pd.read_csv(file_name)
    copy_acc_filtered = um_acc_raw.copy()
    timestamp_flag = 0
    for col in um_acc_raw.columns:
        if "timestamp" in col.lower():
            timestamp_flag = True
        if "timestamp" not in col.lower() and "accel" not in col.lower():
            copy_acc_filtered.drop(col, inplace=True, axis=1)
    if timestamp_flag and len(copy_acc_filtered) >= 4:
        #debug
        print("ok\n")
        um_acc_raw = copy_acc_filtered.filter(items=copy_acc_filtered.columns[:4], axis=1)
        um_acc_raw = um_acc_raw.to_numpy()
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
        timestamp = list(pool.map(parse_date, um_acc_raw[:, 0]))
        um_acc = np.hstack([np.expand_dims(timestamp, axis=1), um_acc_raw[:, 1:]])
        um_acc = um_acc[um_acc[:, 0].argsort()]
        um_acc = sampling_rate(um_acc, 10)
        return um_acc
    else:
        #debug
        if not timestamp_flag:
            print("No timestamp ")
        if len(copy_acc_filtered) <= 4:
            print("Not enough columns")
        if len(copy_acc_filtered) == 0:
            print("Empty")
        print("\n")
        outfile.write(f"Not enough columns or no timestamp. Columns:{copy_acc_filtered.columns}")
        return None


def sampling_rate(data, rate_reduc):
    number_samp = data.shape[0]
    samples_slct = list(range(0,number_samp,rate_reduc))
    new_data = data[samples_slct]
    return np.array(new_data)


def convert_to_date(value):
    return datetime.datetime.strptime(value, '%m/%d/%Y %H:%M:%S.%f')


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

def check_columns(acc_fold: str):
    acc_files = get_accs_files(acc_fold)
    string_col = []
    for file_name in acc_files:
        acc_file = read_acc_file(file_name)
        array = []
        for c in acc_file.columns:
            spl_c = c.split("_")
            array.append("_".join(spl_c[2:]))
        string_col.append("-".join(array))
        #print(f'\n{"-".join(array)}')

    print(np.unique(string_col, return_counts=True))

class Outcomes_ADAPT(Dataset):
    def __init__(self, name, dir_dataset, dir_save, freq = 100, trials_per_file=100000, time_wd=1800, time_drop=900):
        super().__init__(name, dir_dataset, dir_save, freq, trials_per_file)
        self.reduce_rate = 10
        self.time_wd = time_wd * (self.freq/self.reduce_rate)
        self.time_drop = time_drop * (self.freq/self.reduce_rate)
        self.patient_map = self.set_patient_map()
        self.outcomes_file = self.read_labels_file()

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
        patient_enrollment = pd.read_excel('/data/daily_data/patient_id_mapping.xlsx', engine='openpyxl')

        for row in patient_enrollment.itertuples():
            patient_map[row.subject_deiden_id] = row.patient_id

        return patient_map

    def get_accs_files(self):
        accs = []
        for root, dirs, files in os.walk(self.dir_dataset):
            for file in files:
                if file.endswith("SD.csv"):
                    acc_csv = os.path.join(root, file)
                    # just get csv files from Accelerometer directories
                    if fnmatch.fnmatch(root, f'{self.dir_dataset}*/*_Accel/Curated_file/*'):
                        if acc_csv not in accs:
                            accs.append(acc_csv)

        return accs

    def read_labels_file(self):
        df = pd.read_csv(
            '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/PAINandADAPT_clinical_data_outcomes.csv')
        tzdict = {'EST': dateutil.tz.gettz('America/New_York'),
                  'EDT': dateutil.tz.gettz('America/New_York')}
        df['timestamp'] = df['timestamp'].apply(lambda x: dateutil.parser.parse(x + " EST", tzinfos=tzdict))
        return df


    def get_labels(self, patient_id, init_day, last_day):
        df = self.outcomes_file
        df = df[(df['patient_id'] == patient_id) & (df['timestamp'] >= init_day) & (df['timestamp'] <= last_day)]
        df = df.sort_values(by='timestamp')
        pain_measurements = df[(df['pain_score'] != -1)]
        pain_measurements = pain_measurements['timestamp'].dt.to_pydatetime()
        return df, pain_measurements

    def preprocess(self):
        accs_files = self.get_accs_files()
        #accs_files = ["/home/jsenadesouza/DA-healthy2patient/1013_Sensor_Data/I021A/I021A_Accel/Curated_file/2022-03-16_08.55.34_I021A_arm3,4_SD_Session1/I021A_arm3,4_Session1_I021A_arm3,4_Calibrated_SD.csv"]
        trial_id = 1
        output_dir = self.dir_save
        outfile = open('outcomes_acc_PAIN.txt', 'w', buffering=1)

        for file in tqdm(accs_files):
            samples_extracted = 0
            #try:
            if 'wrist' not in file and 'arm' not in file:
                outfile.write(f"\nDiscarding: {file}")
            else:
                file_acc = read_acc_file(file, outfile)
                if file_acc is None:
                    outfile.write(f"\nDiscarding: {file}")
                else:
                    outfile.write(f'\nKeeping: {file}')
                    init_day = np.min(file_acc[:, 0])
                    last_day = np.max(file_acc[:, 0])
                    #print(f'start = {init_day}')
                   #print(f'end = {last_day}')
                    #continue

                    patient_id = file.split("/")[5]

                    outcomes_filtered_df, pain_filtered = self.get_labels(patient_id, init_day, last_day)
                    if len(outcomes_filtered_df) > 0 and len(pain_filtered) > 0:

                        # converting string to datetime
                        #pool = mp.Pool()
                        #file_acc[:, 0] = list(pool.map(convert_to_date, file_acc[:, 0]))
                        acc_ts_list = file_acc[:, 0]

                        for pain_datetime in pain_filtered:

                            idx = bisect_left(acc_ts_list, pain_datetime)
                            idx = idx - 1 if idx >= len(acc_ts_list) else idx
                            # check if there is at least 45 minutes of data before the pain measurement
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
                                        sample = file_acc[start_idx:end_idx]
                                        # add labels
                                        label = process_labels(outcomes_filtered_df, start_ts, end_ts)
                                        if len(label) > 0:
                                            label = "_".join(label.astype(str))
                                            self.add_info_data(label, patient_id, trial_id, sample, output_dir)
                                            trial_id += 1
                                            samples_extracted += 1
            # except:
            #     print("Error on file: ", file)
            if samples_extracted == 0:
                outfile.write(f'\nNo samples extracted for {file}')
            else:
                outfile.write(f'\n{samples_extracted} samples extracted for {file}')

        self.save_data(output_dir)

if __name__ == "__main__":
    outcomesacc_fold = "/home/jsenadesouza/DA-healthy2patient/1013_Sensor_Data/"
    dir_datasets = '/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset_preprocess/'
    outcomes_data = Outcomes_ADAPT('acc_outcomes_ADAPT', outcomesacc_fold, dir_datasets, freq=100, trials_per_file=100)
    outcomes_data.preprocess()