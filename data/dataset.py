import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cupy as cp
import os
import cudf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from bisect import bisect_left
import fnmatch
import dateutil
import pickle
import gc
import dask_cudf


def set_patient_map():
    # create a map between the subject_deiden_id and the patient id
    patient_map = {}
    patient_enrollment = pd.read_excel('/data/daily_data/patient_id_mapping.xlsx', engine='openpyxl')

    for row in patient_enrollment.itertuples():
        patient_map[row.subject_deiden_id] = row.patient_id

    return patient_map


def process_labels(df_data, start_ts, end_ts):
    df = df_data[(df_data['timestamp'] >= start_ts) & (df_data['timestamp'] <= end_ts)]
    df_prev = df_data[df_data['timestamp'] <= (start_ts-np.timedelta64(1, 's'))]
    df = df.to_pandas()
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

        pain_score_prev = df_prev["pain_score"][df_prev["pain_score"] != -1].iloc[-1]
        if pain_score_prev == -1:
            pain_score_prev_class = -1
        elif 0 <= pain_score_prev < 5:
            pain_score_prev_class = "mild"
        else:
            pain_score_prev_class = "severe"
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
             pain_score_prev, pain_score_prev_class,
             sofa_score, sofa_score_class, map, map_class, braden_score, braden_score_class, spo2, spo2_class, cam],
            nan=-1)


def reduce_sampling_rate(data, df_timestamps, reduce_rate):
    data_ts = df_timestamps
    if reduce_rate != 0:
        number_samp = data.shape[0]
        samples_slct = list(range(0, number_samp, int(reduce_rate)))
        new_data = data[samples_slct]
        data_ts = data_ts[samples_slct]
        return new_data, data_ts
    else:
        return data, df_timestamps


def read_acc_file_pain_adapt(file_name, logger):
    df_acc = dask_cudf.read_csv(file_name).compute()
    timestamp_col = None
    for col in df_acc.columns:
        if "timestamp" in col.lower():
            timestamp_col = col
        if "timestamp" not in col.lower() and "accel" not in col.lower():
            df_acc.drop(col, inplace=True, axis=1)
        if "emg" in col.lower():
            raise Exception("EMG data found in accelerometer file")
    if timestamp_col and len(df_acc) >= 4:
        # in case there is more than one accelerometer, drop the others and keep the first one
        columns_to_keep = df_acc.columns[:4].to_numpy()
        diff = set(df_acc.columns.to_numpy()).difference(columns_to_keep)
        for col in diff:
            df_acc.drop(columns=[col], inplace=True)

        # curation already converted the timestamp to EST, so we dont need to convert it again
        # I had to convert timestamps to pandas due to cudf not supporting milliseconds
        # -  that is need to calculate the frequency of the sensor
        # pain and adapt are in EST time zone. But the function above convert it to GMT by default
        timestamps = df_acc[timestamp_col].to_numpy(dtype="datetime64[ns]")
        return timestamps, df_acc.drop(columns=[timestamp_col]).to_cupy()
    else:
        #debug
        if not timestamp_col:
            logger.error("File {}, message: No timestamp ", file_name)
        if len(df_acc) <= 4:
            logger.error("File {}, message: Not enough columns ", file_name)
        if len(df_acc) == 0:
            logger.error("File {}, message: Empty file ", file_name)
        return None


def read_acc_file_intelligenticu(file_name, logger):
    df_acc = cudf.read_csv(file_name, header=10)
    if "Timestamp" not in df_acc.columns[0]:
        logger.error("File {} has no timestamp column".format(file_name))
        return None
    # we are supposing Intelligent ICU is already on GMT timezone, so no need to convert it
    timestamps = cudf.Series(df_acc["Timestamp"], dtype="datetime64[ms]").to_numpy()
    return timestamps, df_acc.drop(columns=["Timestamp"]).to_cupy()


def convert_to_date(value):
    return datetime.strptime(value, '%m/%d/%Y %H:%M:%S.%f')


class PainDataset:
    def __init__(self, dataset_name, dir_dataset, dir_save, trials_per_file, time_wd, time_drop, final_freq, logger):
        self.time_wd = time_wd
        self.final_freq = final_freq
        self.time_drop = time_drop
        self.final_freq = final_freq
        self.logger = logger
        self.time_wd = time_wd
        self.time_drop = time_drop
        self.dir_dataset = dir_dataset
        self.dir_save = dir_save
        self.dataset_name = dataset_name
        self.data = {}
        self.n_pkl = 0
        # when data achieves trials_per_file trials, it will be saved to disk and data will be cleaned
        self.trials_per_file = trials_per_file
        self.patient_map = set_patient_map()
        self.logger = logger
        self.outcomes_file = self.read_labels_file()

    def read_labels_file(self):
        if self.dataset_name == 'intelligent_icu':
            df = cudf.read_csv(
                '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/2016-20119_clinical_data_outcomes.csv')
        else:
            df = cudf.read_csv(
                '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/PAINandADAPT_clinical_data_outcomes.csv')

        # conversion to GMT time
        df['timestamp'] = cudf.Series(
            pd.to_datetime(df["timestamp"].to_pandas()).dt.tz_localize("EST").dt.tz_convert("GMT"))
        return df

    def get_accs_files(self):
        accs = []
        patients = {}
        for root, dirs, files in os.walk(self.dir_dataset):
            for file in files:
                ends_string = "RAW.csv" if self.dataset_name == 'intelligent_icu' else "SD.csv"
                if file.endswith(ends_string):
                    if self.dataset_name == "intelligent_icu":
                        patient = root.split('/')[5].split("_")[1]
                    else:
                        patient = root.split('/')[5]
                    acc_csv = os.path.join(root, file)
                    if patient not in patients:
                        patients[patient] = False
                    # just get csv files from Accelerometer directories
                    if self.dataset_name == "intelligent_icu":
                        path = f'{self.dir_dataset}*/Accelerometer/*'
                    else:
                        path = f'{self.dir_dataset}*/*_Accel/Curated_file/*'

                    if fnmatch.fnmatch(root, path):
                        if 'wrist' in file.lower() or 'arm' in file.lower() or 'emg' in file.lower():
                        #if 'wrist' in file.lower() or 'arm' in file.lower():
                            if 'emg' not in file.lower():
                                patients[patient] = True
                                if acc_csv not in accs:
                                    accs.append(acc_csv)

        for pat, acc_flag in patients.items():
            if not acc_flag:
                self.logger.error("Patient {}, message: no accelerometer data in directory", pat)

        return accs

    def get_labels(self, patient_id, init_day, last_day):
        """
        Filter the outcomes file to get pain measurements
        inside the time range of the accelerometer file

        Returns the filtered dataframe and the pain score column converted to pydatetime
        """
        df = self.outcomes_file
        df = df[(df['patient_id'] == patient_id) & (df['timestamp'] >= init_day) & (df['timestamp'] <= last_day)]
        df = df.sort_values(by='timestamp')
        pain_measurements = df[(df['pain_score'] != -1)]
        pain_measurements = pain_measurements['timestamp']
        return df, pain_measurements

    def add_info_data(self, pain_level, subject, trial_id, trial, output_dir):
        output_name = '{}_s{}_t{}'.format(pain_level.lower(), subject, trial_id)
        self.data[output_name] = trial
        if trial_id % self.trials_per_file == 0 and trial_id != 0:
            self.save_data(output_dir)

    def save_data(self, output_dir):
        try:
            with open(output_dir+self.dataset_name+'_'+str(self.n_pkl)+'.pkl', 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.data = {}
            self.n_pkl += 1
        except Exception as e:
            sys.exit('Error: save pickle {} for {} dataset.\n Error: {}'.format(self.n_pkl, self.dataset_name, e))

    def preprocess(self):
        accs_files = self.get_accs_files()
        trial_id = 1
        output_dir = self.dir_save
        frequencies = []

        for file in tqdm(accs_files):
            try:
                # get accel data and accel timestamps from csv file
                if self.dataset_name == 'intelligent_icu':
                    df_timestamps, acc_cupy = read_acc_file_intelligenticu(file, self.logger)
                else:
                    df_timestamps, acc_cupy = read_acc_file_pain_adapt(file, self.logger)
                # if there is data in the file
                if acc_cupy is not None:
                    # get max and min timestamps to filter the outcomes file - that is made so the search aftewards is faster
                    init_day = df_timestamps.min()
                    last_day = df_timestamps.max()
                    last_day = last_day + np.timedelta64(1, 'D')
                    # get patient id to filter the outcomes file by the current patient
                    # todo fix it when the data changes folder structure
                    patient_id = file.split("/")[5]
                    if self.dataset_name == 'intelligent_icu':
                        patient_id = int(patient_id.split("_")[1])

                    # get outcomes file filtered by patient id and timestamps and pain measurements
                    outcomes_filtered_df, pain_filtered = self.get_labels(patient_id, init_day, last_day)
                    # if there is pain measurements in the time range of the accelerometer file
                    if len(pain_filtered) > 0:
                        samples_extracted = 0
                        # resampled the data to 10Hz
                        freq = 1 / pd.to_timedelta(np.median(np.diff(df_timestamps))).total_seconds()
                        frequencies.append(freq)
                        reduce_rate = freq / self.final_freq
                        acc_cupy_downsampled, acc_ts_list = reduce_sampling_rate(acc_cupy, df_timestamps, reduce_rate)

                        # adjust time-window's reference values to the new frequency
                        # For ICASSP paper we use 15 minutes for both window and drop times.
                        time_wd = self.time_wd * (freq / reduce_rate)
                        time_drop = self.time_drop * (freq / reduce_rate)

                        self.logger.info("File {}, message: Frequency: {}, Reduce rate: {}, Time window: {}", file,
                                         freq, reduce_rate, time_wd)

                        # for each pain measurement in the range of the accelerometer file
                        pain_datetimes = pain_filtered.values_host
                        for i in range(1, len(pain_datetimes)):
                            # get the accel timestamp nearest to the pain measurement
                            # I had to convert acc_ts_list to numpy array because of error when using cudf in bisect function
                            idx = bisect_left(acc_ts_list, pain_datetimes[1])
                            # in case that is too near the end, adjust the index
                            idx = idx - 1 if idx >= len(acc_ts_list) else idx
                            # check if there is at least time_wd + time_drop minutes of data before the pain measurement
                            if idx > time_wd + time_drop:
                                margin_minutes = 5
                                margin = np.timedelta64(margin_minutes, 'm')
                                # check if the accelerometer timestamp is within 5 minutes of the pain measurement
                                if abs(pain_datetimes[1] - acc_ts_list[idx]) <= margin:
                                    # get time_wd minutes before time_drop minutes from the pain measurement
                                    # 15 minutes are dropped because the nurse can be in the room doing some procedures
                                    start_idx = int(idx - time_wd - time_drop)
                                    end_idx = int(idx - time_drop)
                                    ts_sample = acc_ts_list[start_idx:end_idx]

                                    # check if the timestamps in the sample are continuous
                                    if np.max(np.diff(ts_sample)) < np.timedelta64(1, 's'):
                                        # above we filtered the timestamp list, if that range is continuous,
                                        # then the sample is continuous so we can extract it
                                        start_ts = ts_sample[0]

                                        end_ts = ts_sample[-1] + np.timedelta64(int(margin_minutes+time_drop/60/reduce_rate), 'm')
                                        sample = acc_cupy_downsampled[start_idx:end_idx]
                                        # get outcomes data for the sample
                                        label = process_labels(outcomes_filtered_df, start_ts, end_ts)
                                        if len(label) > 0:
                                            label = "_".join(label.astype(str))
                                            # save the sample
                                            self.add_info_data(label, patient_id, trial_id, sample, output_dir)
                                            trial_id += 1
                                            samples_extracted += 1
                                        del sample

                                    del ts_sample
                        if not samples_extracted:
                            self.logger.error("File {}, message: no sample extracted", file)
                        else:
                            self.logger.success("File {}, message: sample extracted", file)
                        del acc_cupy_downsampled
                        del acc_ts_list
                    else:
                        self.logger.error("File {}, message: No pain measurements", file)
                    del outcomes_filtered_df
                    del pain_filtered
                    del init_day
                    del last_day
                else:
                    self.logger.error("File {}, message: No data", file)
                del acc_cupy
                del df_timestamps
                gc.collect()
            except Exception as e:
                self.logger.error("File {}, message: {}", file, e)
        # save the last sample
        self.save_data(output_dir)
        self.logger.info("Frequencies: mean = {}, values = {}", np.mean(frequencies), frequencies)








