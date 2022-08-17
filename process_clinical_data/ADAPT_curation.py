# Author : Jessica

# Modified : Necessary
# Modified on : May 25th
# Modified by : Ziyuan Guan
# Change the curation output file locations


import pandas as pd
import os
import fnmatch
import dateutil
from tqdm import tqdm
from loguru import logger
from multiprocessing import Pool
import multiprocessing


def process_curate_files(curate_files_fold: str):
    _projectID = os.path.basename(curate_files_fold).split('_')[0]
    downtimes_file = pd.read_csv(os.path.join(curate_files_fold, '%s_accel_downtime.csv') % (_projectID))
    downtimes_file['subj_id'] = downtimes_file['subj_id'].str.upper()
    start_end_times_file = pd.read_csv(os.path.join(curate_files_fold, '%s_accel_start_end.csv') % (_projectID))
    start_end_times_file['subj_id'] = start_end_times_file['subj_id'].str.upper()

    return downtimes_file, start_end_times_file


# folder containing the clinical team's .csv files
curate_files_dir = "/data/datasets/ICU_Data/Curation_Files/1013_Sensor_Data"
# root folder containing the accelerometer files
acc_dir = "/data2/datasets/ICU_Data/1013_Sensor_Data"
# folder where the filtered accelerometer files will be saved
output_dir = "/home/jsenadesouza/DA-healthy2patient/1013_Sensor_Data/"
# output_dir = "/data2/datasets/ICU_Data/1013_Sensor_Data"


# read Clinical team's notes
downtimes_file, start_end_times_file = process_curate_files(curate_files_dir)

tzdict = {'EST': dateutil.tz.gettz('America/New_York'),
          'EDT': dateutil.tz.gettz('America/New_York')}


def read_acc_file(file_name: str):
    # read accelerometer file
    acc_file = pd.read_csv(file_name, skiprows=[0, 2], delimiter='\t')
    timestamp_col = acc_file.columns[0]

    if "Shimmer" in timestamp_col:
        raise Exception("Timestamp in a bad format (Shimmer timestamp).")
    else:
        # convert to datetime and adjust timezone to EST
        acc_file[timestamp_col] = pd.to_datetime(acc_file[timestamp_col].astype('int64'), unit='ms')
        acc_file[timestamp_col] = acc_file[timestamp_col].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

    # excludes timestamps above a certain date - used to exclude bad timestamp conversions
    minimum_date = pd.Timestamp(2000, 1, 1)
    minimum_date = minimum_date.tz_localize('US/Eastern')
    if max(acc_file[timestamp_col]) < minimum_date:
        raise Exception(f"File timestamps are out of the project time: {max(acc_file[timestamp_col])}")

    return acc_file


def get_accs_files(dir_dataset: str):
    accs = []
    for root, dirs, files in os.walk(dir_dataset):
        for file in files:
            if file.endswith("SD.csv"):
                acc_csv = os.path.join(root, file)
                # just get csv files from Accelerometer directories
                if fnmatch.fnmatch(root, f'{dir_dataset}/*'):
                    if acc_csv not in accs and "Curated_file" not in root:
                        accs.append(acc_csv)
    return accs


def parse_location(array: list):
    string = array[1].lower()
    if 'leg' in string:
        location = 'ankle'
    elif 'ankle' in string:
        location = 'ankle'
    elif 'arm' in string:
        location = 'arm'
    elif 'emg' in string:
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
        elif 'emg' in string:
            location = 'arm'
        else:
            raise Exception(f"Location {string} dont recognized. String:{array}. Fail")

    return location

def curate_acc(file_name):
    # read the accelerometer file
    try:
        acc_file = read_acc_file(file_name)

        if len(acc_file) > 0:
            timestamp_col = acc_file.columns[0]

            # get patient id and location to filter clinical team's notes
            patient_id = timestamp_col.split("_")[0]
            #check if there is ACC or GYR on the file
            hasmotionsensor = False
            for col in acc_file.columns:
                col = col.lower()
                if "acc" in col or "gyr" in col:
                    hasmotionsensor = True
            if not hasmotionsensor:
                raise Exception("No motion sensor found in the file (no ACC or GYR)")

            location = parse_location(timestamp_col.split("_"))

            # get information regarding this patient and body location on clinical team's notes
            downtimes = downtimes_file[
                (downtimes_file['subj_id'] == patient_id) & (downtimes_file['location'] == location)].values
            start_end_times = start_end_times_file[
                (start_end_times_file['subj_id'] == patient_id) & (
                        start_end_times_file['location'] == location)].values
            if len(start_end_times) == 0:
                raise Exception(f"{patient_id}-{location}")
            else:
                if len(start_end_times[0]) > 2 and start_end_times[0][-2] != "0":
                    start = dateutil.parser.parse(start_end_times[0][-2] + " EST", tzinfos=tzdict)
                else:
                    start = pd.Timestamp.min.tz_localize('US/Eastern')
                if len(start_end_times[0]) > 2 and start_end_times[0][-1] != "0":
                    end = dateutil.parser.parse(start_end_times[0][-1] + " EST", tzinfos=tzdict)
                else:
                    end = (pd.Timestamp.max - pd.Timedelta(days=1)).tz_localize('US/Eastern')

            # filter accelerometer file by the start and end times clinical teams said they put and removed the device
            acc_filtered = acc_file[(acc_file[timestamp_col] > start) & (acc_file[timestamp_col] < end)]

            # filter the accelerometer file by the downtimes the clinical team reported (times when the device is not
            # being worn)
            for downtime_interval in downtimes:
                down_start = downtime_interval[-2]
                down_end = downtime_interval[-1]
                if down_start == '0':
                    down_start = (pd.Timestamp.max - pd.Timedelta(days=1)).tz_localize('US/Eastern')
                if down_end == '0':
                    down_end = pd.Timestamp.min.tz_localize('US/Eastern')
                acc_filtered = acc_filtered[
                    (acc_filtered[timestamp_col] < down_start) | (acc_filtered[timestamp_col] > down_end)]

            copy_acc_filtered = acc_filtered.copy()
            # filter columns that are not timestamp or accel or gyr info
            for col in acc_filtered.columns:
                if "timestamp" not in col.lower() and "accel" not in col.lower() and "gyr" not in col.lower():
                    copy_acc_filtered.drop(col, inplace=True, axis=1)
            if len(copy_acc_filtered) > 1:

                # print(copy_acc_filtered.columns + '\n')
                # save accelerometer file filtered
                _folderName = file_name.split("/")[-2]
                # created extra folder for curated files
                _tempArray = [patient_id, patient_id + '_Accel', 'Curated_file', _folderName]
                output_dir = output_dir + '/' + "/".join(_tempArray)
                out_file_name = os.path.basename(file_name)
                os.makedirs(output_dir, exist_ok=True)
                copy_acc_filtered.to_csv(os.path.join(output_dir, out_file_name))
            else:
                raise Exception(f"Filtered file has no data. Shape: {copy_acc_filtered.shape}")
        else:
            raise Exception("ACC file empty")
    except Exception as e:
        logger.error('    process ACCEL file : {} got error : {}', file_name, e)


def curation():
    logger.add("adapt_loguru.log", enqueue=True)
    # get accelerometer files
    acc_files = get_accs_files(acc_dir)

    n_cpus = multiprocessing.cpu_count()

    with Pool(processes=n_cpus) as p:
        max_ = len(acc_files)
        with tqdm(total=max_) as pbar:
            for _ in p.imap_unordered(curate_acc, acc_files):
                pbar.update()


if __name__ == "__main__":
    curation()
    #curate_acc("/data2/datasets/ICU_Data/1013_Sensor_Data/I037A/I037A_Accel/2022-05-03_11.30.07_I037A_arm1,2_SD_Session1/I037A_arm1,2_Session1_I037A_arm1,2_Calibrated_SD.csv")
