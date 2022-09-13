import os
import fnmatch
import pandas as pd


def check_place_sensor(dir_dataset):
    if "1013" in dir_dataset:
        files = get_accs_files_adapt(dir_dataset, get_folder=True)
    else:
        files = get_accs_files_pain(dir_dataset, get_folder=True)
    for file_path in files:
        arm, ankle = False, False
        file_columns = pd.read_csv(file_path, skiprows=[0, 2], delimiter='\t', nrows=10).columns
        for col in file_columns:
            if "arm" in col:
                arm = True
            if "ankle" in col:
                ankle = True
            if "leg" in col:
                ankle = True
            if "emg" in col:
                arm = True
        if "arm" in file_path.lower():
            arm = True
        if "ankle" in file_path.lower():
            ankle = True
        if "leg" in file_path.lower():
            ankle = True
        if "emg" in file_path.lower():
            arm = True

        if sum([arm, ankle]) != 1:
            print(f"file = {file_path}\n column name: {file_columns[0]}")
            print("\n")



def check_folder_file_name(dir_dataset):
    if "1013" in dir_dataset:
        files = get_accs_files_adapt(dir_dataset, get_folder=True)
    else:
        files = get_accs_files_pain(dir_dataset, get_folder=True)

    for file_path in files:
        if "1013" in dir_dataset:
            external_folder = file_path.split("/")[5]
            middle_folder = file_path.split("/")[6]
            internal_folder = file_path.split("/")[7]
            file_name = file_path.split("/")[-1]
            file_column = pd.read_csv(file_path, skiprows=[0, 2], delimiter='\t', nrows=10).columns[0]
            if external_folder not in middle_folder or \
                external_folder not in internal_folder or \
                external_folder not in file_name or \
                external_folder not in file_column:
                    print(f"file = {file_path}\n column name: {file_column}")
                    print("\n")
        else:
            external_folder = file_path.split("/")[5]
            internal_folder = file_path.split("/")[7]
            file_name = file_path.split("/")[-1]
            file_column = pd.read_csv(file_path, skiprows=[0, 2], delimiter='\t', nrows=10).columns[0]
            if external_folder not in internal_folder or \
                    external_folder not in file_name or \
                    external_folder not in file_column:
                print(f"file = {file_path}\n column name: {file_column}")
                print("\n")


def get_accs_files_pain(dir_dataset: str, get_folder:bool =False):
    accs = []
    for root, dirs, files in os.walk(dir_dataset):
        for file in files:
            if file.endswith("SD.csv"):
                acc_csv = os.path.join(root, file)
                # just get csv files from Accelerometer directories
                if fnmatch.fnmatch(root, f'{dir_dataset}*/Accel/*'):
                    if acc_csv not in accs:
                        if get_folder:
                            accs.append(acc_csv)
                        else:
                            accs.append(acc_csv.split("/")[-1].split(".csv")[0])

    return accs


def get_accs_files_adapt(dir_dataset: str, get_folder: bool = False):
    accs = []
    for root, dirs, files in os.walk(dir_dataset):
        for file in files:
            if file.endswith("SD.csv"):
                acc_csv = os.path.join(root, file)
                # just get csv files from Accelerometer directories
                if fnmatch.fnmatch(root, f'{dir_dataset}*'):
                    if acc_csv not in accs:
                        if get_folder:
                            accs.append(acc_csv)
                        else:
                            accs.append(acc_csv.split("/")[-1].split(".csv")[0])
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


def log_to_csv(curation_log, PROJECT_NAME):
    output_csv = curation_log.replace(".log", ".csv")
    log = open(curation_log)
    files_excluded = []

    df_array = []

    for line in log:
        if PROJECT_NAME + "_Sensor_Data" in line:
            f = line.split("/")[-1].split(".csv")[0]
            files_excluded.append(f)
            patient_id = line.split("/")[-1].split("_")[0]
            error = line.split(", message:")[-1].split("\n")[0]
            df_array.append([patient_id, error, f])

    df = pd.DataFrame(df_array, columns=['patient', 'reason', 'file'])

    df.to_csv(output_csv, index=False)


def check_curation(PROJECT, before, after, curation_log, output_csv):
    if PROJECT == "1013":
        files_before = get_accs_files_adapt(before)
    else:
        files_before = get_accs_files_pain(before)
    files_after = get_accs_files_after(after)

    folder_diff = set(files_before).difference(set(files_after))
    patients_before = []
    for file in files_before:
        patient_id = file.split("_")[0]
        if patient_id not in patients_before:
            patients_before.append(patient_id)

    patients_after = []
    for file in files_after:
        patient_id = file.split("_")[0]
        if patient_id not in patients_after:
            patients_after.append(patient_id)

    pat_diff = set(patients_before).difference(set(patients_after))
    print(f"{len(patients_after)} patients in after\n")
    print(f"{len(patients_before)} patients in before\n")
    print(f"{len(pat_diff)} patients are missing after curation: {pat_diff}")
    log = open(curation_log)
    files_excluded = []

    df_array = []

    for line in log:
        if PROJECT + "_Sensor_Data" in line:
            f = line.split("/")[-1].split(".csv")[0]
            files_excluded.append(f)
            patient_id = line.split("/")[-1].split("_")[0]
            error = line.split("got error : ")[-1].split("\n")[0]
            df_array.append([patient_id, error, f])

    df = pd.DataFrame(df_array, columns=['patient', 'reason', 'file'])

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    PROJECT = "1013"
    before = "/data2/datasets/ICU_Data/1013_Sensor_Data/"
    after = "/home/jsenadesouza/DA-healthy2patient/1013_Sensor_Data/"
    curation_log = "/home/jsenadesouza/DA-healthy2patient/results/curation/adapt_curation.log"
    output_csv = '/home/jsenadesouza/DA-healthy2patient/results/curation/reason_exclusion_curation_ADAPT.csv'
    # PROJECT = "354"
    # before = "/data/datasets/ICU_Data/354_Sensor_Data/"
    # after = "/home/jsenadesouza/DA-healthy2patient/354_Sensor_data/"
    # curation_log = "/home/jsenadesouza/DA-healthy2patient/results/curation/pain_curation.log"
    # output_csv = '/home/jsenadesouza/DA-healthy2patient/results/curation/reason_exclusion_curation_PAIN.csv'
    #check_curation(PROJECT, before, after, curation_log, output_csv)
    #check_folder_file_name("/data2/datasets/ICU_Data/1013_Sensor_Data/")
    #check_folder_file_name("/data/datasets/ICU_Data/354_Sensor_Data/")
    #check_place_sensor("/data2/datasets/ICU_Data/1013_Sensor_Data/")
    #check_place_sensor("/data/datasets/ICU_Data/354_Sensor_Data/")
    log_to_csv("/home/jsenadesouza/DA-healthy2patient/results/outcomes/ADAPT.log", "1013")
