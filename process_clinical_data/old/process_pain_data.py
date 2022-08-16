#%%
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
from utils.clinical_data import read_acc_file, get_accs_files
from bisect import bisect_left, bisect_right


def get_labels(pain_labels, um_acc, patient_id):
    pain_array = []
    # Precompute a list of keys.
    pbar = tqdm(total=len(um_acc))
    for i in range(um_acc.shape[0]):
        matched = False
        acc_timestamp = datetime.strptime(um_acc[i, 0], '%m/%d/%Y %H:%M:%S.%f')
        # search for match
        key = acc_timestamp.date().strftime("%m/%d/%Y") + "_" + patient_id
        if key in pain_labels:
            try:
                timestamps = pain_labels[key]['timestamps']
                r_idx = bisect_right(timestamps, acc_timestamp)
                if r_idx >= len(timestamps):
                    r_idx = len(timestamps) - 1
                right_ts = timestamps[r_idx]
                l_idx = bisect_left(timestamps, acc_timestamp)
                if l_idx >= len(timestamps):
                    l_idx = len(timestamps) - 1
                left_ts = timestamps[l_idx]

                if right_ts == left_ts:
                    if l_idx > 0:
                        l_idx = l_idx - 1
                    left_ts = timestamps[l_idx]


                r,l = float("inf"), float("inf")
                margin = timedelta(minutes=45)
                if right_ts.date() <= acc_timestamp.date() <= right_ts.date():
                    r = right_ts - acc_timestamp
                    if r <= margin:
                        matched = True
                if left_ts.date() <= acc_timestamp.date() <= left_ts.date():
                    l = left_ts - acc_timestamp
                    if r <= margin:
                        matched = True

                if matched:
                    measured_now = 0
                    if r < l:
                        pain = pain_labels[key]['pain_score'][r_idx]
                        if r <= timedelta(minutes=5):
                            measured_now = 1
                    else:
                        pain = pain_labels[key]['pain_score'][l_idx]
                        if l <= timedelta(minutes=5):
                            measured_now = 1

                    pain_array.append({'x': um_acc[i, 1], 'y': um_acc[i, 2], 'z': um_acc[i, 3],
                                                                'patiend_id': patient_id, 'timestamp': acc_timestamp,
                                                                'pain_score': pain, 'measured_now': measured_now})
            except:
                print(l_idx)
                print(r_idx)


        pbar.update(1)
    pbar.close()
    return pain_array

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


def get_pain_labels():
    pain_path = '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/pain_data.npz'
    pain_file = np.load(pain_path, allow_pickle=True)['pain']
    pain_dict = {}
    for row in pain_file:
        pain_datetime = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S.%f')
        pain_date = pain_datetime.date().strftime("%m/%d/%Y")
        if pain_date + "_" + row[1] in pain_dict:
            pain_dict[pain_date + "_" + row[1]]['timestamps'].append(pain_datetime)
            pain_dict[pain_date + "_" + row[1]]['pain_score'].append(row[2])
        else:
            pain_dict[pain_date + "_" + row[1]] = {'timestamps': [pain_datetime], 'pain_score': [row[2]]}

    return pain_dict

accs_files = get_accs_files()
pain_labels = get_pain_labels()

for file in accs_files.keys():

    patient_id = file.split("/")[5].split("_")[1]
    init_day = datetime.strptime(file.split("/")[-2].split(" ")[1], '%m.%d.%y')
    last_day = datetime.strptime(file.split("/")[-2].split(" ")[3], '%m.%d.%y')
    current = init_day
    match = False
    pain_filtered = {}
    while last_day >= current:
        key = current.date().strftime("%m/%d/%Y") + "_" + patient_id
        if key in pain_labels:
            pain_filtered[key] = pain_labels[key]
            match = True
        current = current + timedelta(days=1)

    if match:
        print(file)

        um_acc = read_acc_file(file)
        pain_array = get_labels(pain_filtered, um_acc, patient_id)
        df = pd.DataFrame(pain_array, columns=['x', 'y', 'z', 'patiend_id', 'timestamp', 'pain_score', 'measured_now'])

        folder = '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/pain_acc/'
        out_name = f"{folder}pain30minmarginmeasurednowcol_p_{accs_files[file]['patient']}_t_{accs_files[file]['timestamp']}_bp_{accs_files[file]['bodypart']}"
        df.to_csv(out_name + ".csv", index=False)

    else:
        print("NOT MATCHED: " + file)
