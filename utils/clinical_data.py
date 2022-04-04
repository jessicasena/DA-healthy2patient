import os
import pandas as pd
import numpy as np

def get_accs_files():
    path = "/data/datasets/ICU_Data/Sensor_Data/Patient_60/Accelerometer/"
    accs = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("RAW.csv"):
                patient = root.split('/')[5].split("_")[1]
                timestamp = file.split("(")[-1].split(")")[0]
                bodypart = file.split("_")[0]
                acc_csv = os.path.join(root, file)
                #if os.path.isdir(os.path.join(path, f"Patient_{patient}", "Activity", "Sampled_Images")):
                if acc_csv not in accs:
                    accs[acc_csv] = {'patient': patient, 'timestamp': timestamp, "bodypart": bodypart}
    return accs


def read_acc_file(file_name):
    um_acc_raw = pd.read_csv(file_name, header=10)
    um_acc_raw = um_acc_raw.to_numpy()
    um_acc = np.concatenate((um_acc_raw, np.zeros((um_acc_raw.shape[0], 1))), axis=1)
    return um_acc