import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dataset.Datasets import Dataset
import numpy as np
import glob, os
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


class SignalsIHealth(Enum):
    timestamp = 0
    acc1_dominant_wrist_X = 1
    acc1_dominant_wrist_Y = 2
    acc1_dominant_wrist_Z = 3
    activityID = 4


def relabel(label):
    if label == -1.0:
        return "unknown"
    elif label == 0.0:
        return "inbed"
    elif label == 1.0:
        return "notinbed"


class IHealth(Dataset):
    def print_info(self):
        return """
                device: IMU
                frequency: 100Hz
                positions: dominant wrist, chest and dominant side's ankle
                sensors: heart rate, temperature, acc, gyr and mag
                """

    def preprocess(self):
        files = glob.glob(pathname=os.path.join(self.dir_dataset, '*.csv'))
        output_dir = self.dir_save
        trial_id = 1
        delta = timedelta(minutes=1)
        print(f'processing = {self.dir_dataset}', flush=True)
        for f in tqdm(files, desc="files", position=0):
            subject_id = int(f.split('/')[-1].split('_')[1])
            df = pd.read_csv(f, header=None)
            # filter unannotated rows
            df = df.loc[df[4] >= 0.0]
            if not len(df):
                print(f'\nHas no annotated data: \n{f}', flush=True)
            else:
                current_label = df.iloc[0, 4]
                timestamp_previous = datetime.strptime(df.iloc[0, 0], '%m/%d/%Y %H:%M:%S.%f')
                trial = []
                #for i in tqdm(range(len(df)), desc="lines", position=1):

                for row in tqdm(df.itertuples(index=False), desc="lines", position=1, total=df.shape[0]):
                    timestamp = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S.%f')
                    label = relabel(row[4])

                    if label != current_label or abs(timestamp - timestamp_previous) > delta:
                        self.add_info_data(str(current_label), subject_id, trial_id, trial, output_dir)
                        trial_id += 1
                        trial = []
                    current_label = label
                    trial.append(row[1:4])
                    timestamp_previous = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S.%f')

        self.save_data(output_dir)