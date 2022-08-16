import pandas as pd
import os
import fnmatch
import sys
import dateutil
from tqdm import tqdm


def read_acc_file(file_name: str):
    # read accelerometer file
    acc_file = pd.read_csv(file_name, skiprows=[0, 2], delimiter='\t', index_col=0)
    timestamp_col = acc_file.columns[0]

    if "Shimmer" in timestamp_col:
        acc_file = []
    else:
        # convert to datetime and adjust timezone to EST
        acc_file[timestamp_col] = pd.to_datetime(acc_file[timestamp_col].astype('int64'), unit='ms')
        acc_file[timestamp_col] = acc_file[timestamp_col].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

    return acc_file


def get_accs_files(dir_dataset: str):
    accs = []
    for root, dirs, files in os.walk(dir_dataset):
        for file in files:
            if file.endswith("SD.csv"):
                acc_csv = os.path.join(root, file)
                # just get csv files from Accelerometer directories
                if fnmatch.fnmatch(root, f'{dir_dataset}*/Accel/*'):
                    if acc_csv not in accs:
                        accs.append(acc_csv)

    return accs


def process_curate_files(curate_files_fold: str):
    end_reason_file = pd.read_csv(os.path.join(curate_files_fold, 'end_reason_time.csv'))
    end_reason_file['subj_id'] = end_reason_file['subj_id'].str.upper()
    downtimes_file = pd.read_csv(os.path.join(curate_files_fold, 'patients_downtime.csv'))
    downtimes_file['subj_id'] = downtimes_file['subj_id'].str.upper()
    start_end_times_file = pd.read_csv(os.path.join(curate_files_fold, 'server_accel_times.csv'))
    start_end_times_file['subj_id'] = start_end_times_file['subj_id'].str.upper()

    return end_reason_file, downtimes_file, start_end_times_file


def parse_location(array: list):
    string = array[1].lower()
    if 'leg' in string:
        location = 'ankle'
    elif 'ankle' in string:
        location = 'ankle'
    elif 'arm' in string:
        location = 'arm'
    else:
        print(f"\nLocation {string} dont recognized. Trying {array[2]}")
        string = array[2].lower()
        if 'leg' in string:
            location = 'ankle'
        elif 'ankle' in string:
            location = 'ankle'
        elif 'arm' in string:
            location = 'arm'
        else:
            sys.exit(f"\nLocation {string} dont recognized. String:{array}. Fail")

    return location


def curate_acc(acc_fold: str, curate_files_fold: str, output: str):
    # get açç tje accelerometer files
    acc_files = get_accs_files(acc_fold)
    # read Clinical team's notes
    end_reason_file, downtimes_file, start_end_times_file = process_curate_files(curate_files_fold)

    tzdict = {'EST': dateutil.tz.gettz('America/New_York'),
              'EDT': dateutil.tz.gettz('America/New_York')}


    for file_name in tqdm(acc_files):
        # read the accelerometer file
        acc_file = read_acc_file(file_name)

        if len(acc_file) > 0:
            timestamp_col = acc_file.columns[0]

            # get patient id and location to filter clinical team's notes
            patient_id = timestamp_col.split("_")[0]
            location = parse_location(timestamp_col.split("_"))

            # get information regarding this patient and body location on clinical team's notes
            end_reason = end_reason_file[end_reason_file['subj_id'] == patient_id].values[0][-1]
            if not pd.isna(end_reason):
                end_reason = dateutil.parser.parse(end_reason + " EST", tzinfos=tzdict)
            downtimes = downtimes_file[
                (downtimes_file['subj_id'] == patient_id) & (downtimes_file['location'] == location)].values
            start_end_times = start_end_times_file[
                (start_end_times_file['subj_id'] == patient_id) & (start_end_times_file['location'] == location)].values[0]
            start = dateutil.parser.parse(start_end_times[-2] + " EST", tzinfos=tzdict)
            end = dateutil.parser.parse(start_end_times[-1] + " EST", tzinfos=tzdict)
            if not pd.isna(end_reason):
                if end > end_reason:
                    end = end_reason

            # filter accelerometer file by the start and end times clinical teams said they put and removed the device
            acc_filtered = acc_file[(acc_file[timestamp_col] > start) & (acc_file[timestamp_col] < end)]

            # filter the accelerometer file by the downtimes the clinical team reported (times when the device is not
            # being worn)
            for downtime_interval in downtimes:
                down_start = downtime_interval[-2]
                down_end = downtime_interval[-1]
                acc_filtered = acc_filtered[(acc_filtered[timestamp_col] < down_start) | (acc_filtered[timestamp_col] > down_end)]


            copy_acc_filtered = acc_filtered.copy()
            # filter columns that are not timestamp or accel or gyr info
            for col in acc_filtered.columns:
                if "timestamp" not in col.lower() and "accel" not in col.lower() and "gyr" not in col.lower():
                    copy_acc_filtered.drop(col, inplace=True, axis=1)

            print(copy_acc_filtered.columns + '\n')
            # save accelerometer file filtered
            output_dir  = output + "/".join(file_name.split("/")[5:-1])
            out_file_name = file_name.split("/")[-1]
            os.makedirs(output_dir, exist_ok=True)
            copy_acc_filtered.to_csv(os.path.join(output_dir, out_file_name))

if __name__ == "__main__":
    # # root folder containing the accelerometer files
    acc_dir = "/data/datasets/ICU_Data/354_Sensor_Data/"
    # # folder containing the clinical team's .csv files
    # curate_files_dir = "/home/jsenadesouza/DA-healthy2patient/PAIN_downtimes/"
    # # folder where the filtered accelerometer files will be saved
    # output_dir = "/home/jsenadesouza/DA-healthy2patient/354_Sensor_data/"
    # curate_acc(acc_dir, curate_files_dir, output_dir)


    import os
    import numpy as np
    count = 0
    patients = []
    for root, dirs, files in os.walk("/home/jsenadesouza/DA-healthy2patient/354_Sensor_data/"):
        for file in files:
            if file.endswith("SD.csv"):
                acc_csv = os.path.join(root, file)
                # just get csv files from Accelerometer directories
                if fnmatch.fnmatch(root, f'{acc_dir}*/Accel/*'):
                    patient = root.split("/")[5]
                    acc_file = pd.read_csv(acc_csv, skiprows=[0, 2], nrows=5, delimiter='\t')
                    intern_patient = acc_file.columns[0].split("_")[0]
                    if patient != intern_patient:
                        print(acc_csv)
                        print(f"Name on folder = {patient}, Name inside table = {intern_patient}")

            # if file_size < 1024:
            #     print(file)
            #     count += 1
                # patients.append(file.split("_")[0])
                # file_name_ori = os.path.join(acc_dir + root.split("354_Sensor_data/")[1], file)
                # acc_file = pd.read_csv(file_name_ori, skiprows=[0,2], nrows=5, delimiter='\t')
                # timestamp_col = acc_file.columns[0]

                # convert to datetime and adjust timezone to EST
                #acc_file[timestamp_col] = pd.to_datetime(acc_file[timestamp_col].astype('int64'), unit='ms')
                #acc_file[timestamp_col] = acc_file[timestamp_col].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                # print(acc_file[timestamp_col].astype('int64'))
