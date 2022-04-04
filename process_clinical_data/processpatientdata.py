#!/usr/bin/env python
# coding: utf-8

# Ignore (have no samples)
# 
#     - agitated_elevating_the_head
#     - agitated_tilting_the_bed
#     - quietly_elevating_the_head_of_the_bed
#     - quietly_tilting_the_bed
#     - tilting_the_bed
#     - sitting_in_chair    
# 
# Use:
# 
#     Up/walking/standing:
#     - assisted_1_person_assisting 280
#     - assisted_2_person_assisting 515
#     - assisted_more_than_2_person_assisting 38
#     - standing_up 1833
#     - in_an_upright_position 4010
#     Using Walker:
#     - assisted_with_walker 427
#     - in_a_walker 429
#     Using Wheelchair
#     - assisted_with_wheelchair 36
#     - in_a_wheelchair 458
#     laying on bed:
#     - elevating_the_head_of_the_bed 572369
#     - tilting_the_bed 3398
#     Sitting
#     - in_a_chair 17981
#     - lying_in_chair 2912
#     - on_the_edge_of_the_bed 1342
#     Unclear
#     - unclear 186868
#     
#     
#     Exercices
#      - exercises_as_determined_by_physical_therapy_occupational_therapy 230
#     
#     
#     patient_not_in_bed 16935

# In[1]:


import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from statistics import multimode
from tqdm import tqdm
from bisect import bisect_left, bisect_right

# In[2]:


#get_ipython().system('jupyter nbextension enable --py widgetsnbextension')


# In[3]:


def get_accs_files():
    path = "/data/datasets/ICU_Data/Sensor_Data/"
    accs = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("RAW.csv"):
                patient = root.split('/')[5].split("_")[1]
                timestamp = file.split("(")[-1].split(")")[0]
                bodypart = file.split("_")[0]
                acc_csv = os.path.join(root, file)
                if os.path.isdir(os.path.join(path, f"Patient_{patient}", "Activity", "Sampled_Images")):
                    if acc_csv not in accs:
                        accs[acc_csv] = {'patient': patient, 'timestamp': timestamp, "bodypart": bodypart}
    return accs


# In[4]:


def read_annotation_file():
    ann = pd.read_csv("/home/jsenadesouza/Activity_Annotation_Agreement_2019-11-23.csv")
    ts_ann = ann.filter(items=["image_name", "patient_not_in_bed"]).to_numpy()
    return ts_ann


# In[5]:


def get_timestamp_labels(ts_ann):
    timestamp_labels = {}
    ts_teste = 0
    for item in ts_ann:
        timestamp_size = len(item[0].split("-"))
        if timestamp_size == 6:
            if item[0].split("-")[4]:
                ts = datetime.strptime(item[0], '%Y-%m-%d-%H%M-%S-00%f')
            else:
                ts = datetime.strptime(item[0], '%Y-%m-%d-%H%M--00%f')
        elif timestamp_size == 7:
            ts = datetime.strptime(item[0], '%Y-%m-%d-%H-%M-%S-00%f')
        elif timestamp_size == 8:
            ts = datetime.strptime(item[0], '%Y-%m-%d-%H-%M-%S-00%f')
        else:
            print(item[0])
        if ts not in timestamp_labels:
            timestamp_labels[ts] = [item[1]]
        else:
            timestamp_labels[ts].append(item[1])
    return timestamp_labels


# In[6]:


def read_acc_file(file_name):
    um_acc_raw = pd.read_csv(file_name, header=10)
    um_acc_raw = um_acc_raw.to_numpy()
    um_acc = np.concatenate((um_acc_raw, np.zeros((um_acc_raw.shape[0], 1))), axis=1)
    return um_acc


# In[7]:


def get_labels(timestamp_labels, um_acc):
    # Precompute a list of keys.
    list_timestamps = sorted(list(timestamp_labels.keys()))
    delta = timedelta(minutes=1)
    pbar = tqdm(total=len(um_acc))
    distances = []

    for i in range(um_acc.shape[0]):
        matched = False
        acc_timestamp = datetime.strptime(um_acc[i, 0], '%m/%d/%Y %H:%M:%S.%f')
        # search for match
        right_ts = list_timestamps[bisect_right(list_timestamps, acc_timestamp)]
        left_ts = list_timestamps[bisect_left(list_timestamps, acc_timestamp)]
        if right_ts == left_ts:
            left_ts = list_timestamps[bisect_left(list_timestamps, acc_timestamp) - 1]

        labels = []
        if (right_ts - delta) <= acc_timestamp <= (right_ts + delta):
            labels.extend(timestamp_labels[right_ts])
            matched = True
        if (left_ts - delta) <= acc_timestamp <= (left_ts + delta):
            labels.extend(timestamp_labels[left_ts])
            matched = True
        if matched:
            labels_voting = multimode(labels)
            um_acc[i, -1] = labels_voting[0] if len(labels_voting) == 1 else -2
        else:
            distances.append(
                abs(acc_timestamp - left_ts) if abs(acc_timestamp - left_ts) < abs(acc_timestamp - right_ts) else abs(
                    acc_timestamp - right_ts))
            um_acc[i, -1] = -1

        pbar.update(1)
    pbar.close()
    return um_acc, distances


# In[ ]:


accs_files = get_accs_files()
ts_ann = read_annotation_file()
timestamp_labels = get_timestamp_labels(ts_ann)
statistics_cvs = pd.DataFrame(columns=['not_matched', 'conflict_ann', 'inbed', 'outbed', 'distance'])
for file in accs_files.keys():
    print(file)
    um_acc = read_acc_file(file)
    final_acc_file, distances = get_labels(timestamp_labels, um_acc)

    not_matched = np.count_nonzero(final_acc_file[:, -1] == -1)
    conflict_ann = np.count_nonzero(final_acc_file[:, -1] == -2)
    inbed = np.count_nonzero(final_acc_file[:, -1] == 0)
    outbed = np.count_nonzero(final_acc_file[:, -1] == 1)
    try:
        mean_dist = np.mean(distances)
    except:
        mean_dist = -1

    new_row = pd.DataFrame([[not_matched, conflict_ann, inbed, outbed, mean_dist]],
                           columns=['not_matched', 'conflict_ann', 'inbed', 'outbed', 'distance'])
    statistics_cvs = pd.concat([statistics_cvs, new_row], sort=False, ignore_index=True)
    print(f"Not Matched: {not_matched}, {(not_matched / len(final_acc_file)) * 100:.4}%")
    print(f"Statistics: {np.unique(final_acc_file[:, -1], return_counts=True)}")
    print(f"Average distance: {mean_dist}")

    out_name = f"p_{accs_files[file]['patient']}_t_{accs_files[file]['timestamp']}_bp_{accs_files[file]['bodypart']}"
    DF = pd.DataFrame(final_acc_file)
    DF.to_csv(out_name + ".csv", header=False, index=False)
    statistics_cvs.to_csv("statistics_partial.csv", index=False)
statistics_cvs.to_csv("statistics.csv", index=False)

# In[ ]:


# !jupyter nbconvert --to script Untitled.ipynb

