#%%
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from statistics import multimode
from tqdm import tqdm
from bisect import bisect_left, bisect_right


# In[7]:


def get_labels(sofa_labels, um_acc, patient_id):
    sofa_array = []
    # Precompute a list of keys.
    pbar = tqdm(total=len(um_acc))
    for i in range(um_acc.shape[0]):
        acc_timestamp = datetime.strptime(um_acc[i, 0], '%m/%d/%Y %H:%M:%S.%f')
        # search for match
        key = acc_timestamp.date().strftime("%Y-%m-%d") + "_" + patient_id
        if key in sofa_labels:
            sofa_array.append({'x': um_acc[i, 1], 'y': um_acc[i, 2], 'z': um_acc[i, 3],
                                                            'patiend_id': patient_id, 'timestamp': acc_timestamp,
                                                            'sofa_score': sofa_labels[key]})

        pbar.update(1)
    pbar.close()
    return sofa_array


def get_sofa_labels():
    sofa_path = '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/sofa_final.npz'
    sofa_file = np.load(sofa_path, allow_pickle=True)['sofa']
    sofa_dict = {}
    for row in sofa_file:
        sofa_dict[row[0] + "_" + row[1]] = row[2]

    return sofa_dict


accs_files = get_accs_files()
sofa_labels = get_sofa_labels()

for file in accs_files.keys():

    patient_id = file.split("/")[5].split("_")[1]
    init_day = datetime.strptime(file.split("/")[-2].split(" ")[1], '%m.%d.%y')
    last_day = datetime.strptime(file.split("/")[-2].split(" ")[3], '%m.%d.%y')
    current = init_day
    match = False
    sofa_filtered = {}
    while last_day >= current:
        key = current.date().strftime("%Y-%m-%d") + "_" + patient_id
        if key in sofa_labels:
            sofa_filtered[key] = sofa_labels[key]
            match = True
        current = current + timedelta(days=1)

    if match:
        print(file)

        um_acc = read_acc_file(file)
        sofa_array = get_labels(sofa_filtered, um_acc, patient_id)
        df = pd.DataFrame(sofa_array, columns=['x', 'y', 'z', 'patiend_id', 'timestamp', 'sofa_score'])

        folder = '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/sofa_acc/'
        out_name = f"{folder}sofa_p_{accs_files[file]['patient']}_t_{accs_files[file]['timestamp']}_bp_{accs_files[file]['bodypart']}"
        df.to_csv(out_name + ".csv", index=False)

    else:
        print("NOT MATCHED: " + file)
