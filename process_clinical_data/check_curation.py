import os
import fnmatch
import pandas as pd

def get_accs_files_before(dir_dataset: str):
    accs = []
    for root, dirs, files in os.walk(dir_dataset):
        for file in files:
            if file.endswith("SD.csv"):
                # just get csv files from Accelerometer directories
                if fnmatch.fnmatch(root, f'{dir_dataset}*/Accel/*'):
                    if file not in accs:
                        accs.append(file.split(".csv")[0])

    return accs

def get_accs_files_after(dir_dataset: str):
    accs = []
    for root, dirs, files in os.walk(dir_dataset):
        for file in files:
            if file.endswith("SD.csv"):
                # just get csv files from Accelerometer directories
                if fnmatch.fnmatch(root, f'{dir_dataset}*/*/Curated_file/*'):
                    if file not in accs:
                        accs.append(file.split(".csv")[0])

    return accs



before = "/data/datasets/ICU_Data/354_Sensor_Data/"
after = "/home/jsenadesouza/DA-healthy2patient/354_Sensor_data/"

files_before = get_accs_files_before(before)
files_after = get_accs_files_after(after)

folder_diff = set(files_before).difference(set(files_after))

log = open("/home/jsenadesouza/DA-healthy2patient/code/process_clinical_data/pain_loguru.log")
files_excluded = []

df_array = []
for line in log:
    if "354_Sensor_Data" in line:
        f = line.split("/")[-1].split(".csv")[0]
        files_excluded.append(f)
        patient_id = line.split("/")[-1].split("_")[0]
        error = line.split("got error : ")[-1].split("\n")[0]
        df_array.append([patient_id, error, f])

df = pd.DataFrame(df_array, columns = ['patient','reason','file'])

df.to_csv('reason_exclusion_curation_PAIN.csv', index=False)

ok = 0