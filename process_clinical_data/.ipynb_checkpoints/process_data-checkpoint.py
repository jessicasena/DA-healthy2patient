"""
Read clinical data outcomes files and process them.
Merge information into one file.
- lenght of stay
- vitals
- pulse ox
- mobility
- medications - opioids x not
- diagnoses
- braden score
- cam - for delirium - if its moving a lot or not
- sofa score
- pain score
- blood pressure
- mortality - if the patient died or not

"""

import os
import pandas as pd
from datetime import datetime, timedelta


def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return df.loc[exactmatch.index[0]]
    else:
        lower = df[df[colname] < value][colname]
        if len(lower) > 0:
            lowerneighbour_ind = lower.idxmax()
            lowerneighbour = df.loc[lowerneighbour_ind]
        else:
            lowerneighbour = df.iloc[0]
        upper = df[df[colname] > value][colname]
        if len(upper) > 0:
            upperneighbour_ind = upper.idxmin()
            upperneighbour = df.loc[upperneighbour_ind]
        else:
            upperneighbour = df.iloc[-1]
        if abs(lowerneighbour[colname] - value) <= abs(value - upperneighbour[colname]):
            return lowerneighbour
        else:
            return upperneighbour

def fill_missing_values(df, colname, hours_margin):
    missing = df[df[colname] == -1]
    complete = df[df[colname] != -1]
    before = len(missing)

    for index, row in missing.iterrows():
        nearest_row = find_neighbours(row['timestamp'], complete, 'timestamp')
        if abs(nearest_row["timestamp"] - row["timestamp"]) < timedelta(hours=hours_margin):
            missing.at[index, colname] = nearest_row[colname]

    after = len(missing[missing[colname] == -1])

    print(f"{colname} - {hours_margin} hours margin = missing before: {before} missing after: {after}")

    df = pd.concat([missing, complete])
    return df


def process_vitals(dir, outcomes):
    vitals = pd.read_csv(os.path.join(dir, 'vitals_0_trimmed.csv'))

    for index, row in vitals.iterrows():
        # standardize the timestamp
        timestamp = datetime.strptime(row['vitals_datetime'], '%Y-%m-%d %H:%M:%S')
        timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
        key = f"{timestamp}_{row['record_id']}"
        if not pd.isna(row['heart_rate']):
            hr = float(row['heart_rate'])
        else:
            hr = -1
        if not pd.isna(row['temp_farenheit']):
            temp = float(row['temp_farenheit'])
        else:
            temp = -1

        if key in outcomes:
            outcomes[key]['heart_rate'] = hr
            outcomes[key]['temp_farenheit'] = temp
        else:
            outcomes[key] = {'heart_rate': hr, 'temp_farenheit': temp}
    return outcomes


def process_encounters(dir, outcomes):
    """
    Process lenght of stay files.
    """
    vitals = pd.read_csv(os.path.join(dir, 'encounters_0_trimmed.csv'))

    for index, row in vitals.iterrows():
        # standardize the timestamp
        # timestamp_patient_id
        key = str(row['record_id'])
        admit = datetime.strptime(row['admit_datetime'], '%Y-%m-%d %H:%M:%S')
        dischg = datetime.strptime(row['dischg_datetime'], '%Y-%m-%d %H:%M:%S')
        lenght_stay = abs((dischg - admit).days)
        is_dead = 1 if type(row['death_date']) == str else 0
        if key in outcomes:
            outcomes[key]['lenght_stay'] = lenght_stay
            outcomes[key]['is_dead'] = is_dead
        else:
            outcomes[key] = {'lenght_stay': lenght_stay, 'is_dead': is_dead}
    return outcomes


def process_painscore(dir, outcomes):
    """
    Process lenght of stay files.
    """
    pain = pd.read_csv(os.path.join(dir, 'pain_0_trimmed.csv'))

    for index, row in pain.iterrows():
        # standardize the timestamp
        # timestamp_patient_id
        timestamp = datetime.strptime(row['pain_datetime'], '%Y-%m-%d %H:%M:%S')
        timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
        key = f"{timestamp}_{row['record_id']}"
        try:
            score = int(row['pain_uf_dvprs'])
        except:
            score = -1

        if key in outcomes:
            outcomes[key]['pain_score'] = score
        else:
            outcomes[key] = {'pain_score': score}
    return outcomes


def process_sofascore(dir, outcomes):
    """
    Process lenght of stay files.
    """
    sofa = pd.read_csv(os.path.join(dir, 'sofa_0_trimmed.csv'))

    for index, row in sofa.iterrows():
        # standardize the timestamp
        # timestamp_patient_id
        timestamp = datetime.strptime(row['date_of_care'], '%Y-%m-%d')
        timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
        key = f"{timestamp}_{row['record_id']}"
        try:
            score = int(row['sofa_score'])
        except:
            score = -1

        if key in outcomes:
            outcomes[key]['sofa_score'] = score
        else:
            outcomes[key] = {'sofa_score': score}
    return outcomes


def process_blood_pressure(dir, outcomes):
    """
    Process lenght of stay files.
    """
    bp = pd.read_csv(os.path.join(dir, 'blood_pressure_0_trimmed.csv'))

    for index, row in bp.iterrows():
        # standardize the timestamp
        # timestamp_patient_id
        timestamp = datetime.strptime(row['bp_datetime'], '%Y-%m-%d %H:%M:%S')
        timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
        key = f"{timestamp}_{row['record_id']}"

        if not pd.isna(row["map_a_line"]):
            score = int(row['map_a_line'])
        elif not pd.isna(row["map_non_invasive"]):
            score = int(row['map_non_invasive'])
        else:
            score = -1

        if key in outcomes:
            outcomes[key]['blood_pressure'] = score
        else:
            outcomes[key] = {'blood_pressure': score}
    return outcomes


def process_braden(dir, outcomes):
    """
    Process lenght of stay files.
    """
    braden = pd.read_csv(os.path.join(dir, 'braden_0_trimmed.csv'))

    for index, row in braden.iterrows():
        # standardize the timestamp
        # timestamp_patient_id
        timestamp = datetime.strptime(row['braden_datetime'], '%Y-%m-%d %H:%M:%S')
        timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
        key = f"{timestamp}_{row['record_id']}"
        try:
            score = float(row['braden_score'])
        except:
            score = -1

        if key in outcomes:
            outcomes[key]['braden_score'] = score
        else:
            outcomes[key] = {'braden_score': score}
    return outcomes


def process_pulse_ox(dir, outcomes):
    """
    Process lenght of stay files.
    """
    spo2 = pd.read_csv(os.path.join(dir, 'respiratory_0_trimmed.csv'))

    for index, row in spo2.iterrows():
        # standardize the timestamp
        # timestamp_patient_id
        timestamp = datetime.strptime(row['respiratory_datetime'], '%Y-%m-%d %H:%M:%S')
        timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
        key = f"{timestamp}_{row['record_id']}"
        try:
            score = float(row['spo2'])
        except:
            score = -1
        if key in outcomes:
            outcomes[key]['spo2'] = score
        else:
            outcomes[key] = {'spo2': score}
    return outcomes


def process_cam(dir, outcomes):
    """
    Process lenght of stay files.
    """
    cam = pd.read_csv(os.path.join(dir, 'cam_0_trimmed.csv'))

    for index, row in cam.iterrows():
        # standardize the timestamp
        # timestamp_patient_id
        timestamp = datetime.strptime(row['recorded_time'], '%Y-%m-%d %H:%M:%S')
        timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
        key = f"{timestamp}_{row['record_id']}"
        if row['meas_value'] == 'Negative':
            score = 0
        elif row['meas_value'] == 'Positive':
            score = 1
        else:
            score = -1
        if key in outcomes:
            outcomes[key]['cam'] = score
        else:
            outcomes[key] = {'cam': score}
    return outcomes


def process_1619_outcomes(outcomes_dir, output_dir):
    """
    Process 1619 outcomes files.
    """
    outcomes = {}
    outcomes_all = {}
    outcomes = process_vitals(outcomes_dir, outcomes)
    outcomes = process_painscore(outcomes_dir, outcomes)
    outcomes = process_sofascore(outcomes_dir, outcomes)
    outcomes = process_blood_pressure(outcomes_dir, outcomes)
    outcomes = process_braden(outcomes_dir, outcomes)
    outcomes = process_pulse_ox(outcomes_dir, outcomes)
    outcomes = process_cam(outcomes_dir, outcomes)
    outcomes_all = process_encounters(outcomes_dir, outcomes_all)

    dict_list = []
    for key, value in outcomes.items():
        patient_id = key.split('_')[1]
        timestamp = datetime.strptime(key.split('_')[0], '%m-%d-%Y %H:%M')
        dict = {'timestamp': timestamp, 'patient_id': patient_id}
        dict.update(value)
        dict.update(outcomes_all[patient_id])
        dict_list.append(dict)

    df = pd.DataFrame(dict_list,
                      columns=['timestamp', 'patient_id', 'heart_rate', 'temp_farenheit', 'lenght_stay', 'is_dead',
                               'pain_score', 'sofa_score', 'blood_pressure', 'braden_score', 'spo2', 'cam'])
    df = df.fillna(-1)
    # interpolate missing values
    df = fill_missing_values(df, 'heart_rate', 2)
    df = fill_missing_values(df, 'temp_farenheit', 2)
    df = fill_missing_values(df, 'pain_score', 4)
    df = fill_missing_values(df, 'sofa_score', 24)
    df = fill_missing_values(df, 'blood_pressure', 2)
    df = fill_missing_values(df, 'braden_score', 24)
    df = fill_missing_values(df, 'spo2', 2)
    df = fill_missing_values(df, 'cam', 12)

    df.to_csv(os.path.join(output_dir, '2016-20119_clinical_data_outcomes.csv'), index=False)

if __name__ == '__main__':
    input_dir = '/data/datasets/ICU_Data/EHR_Data/truncated/2020-02-26/'
    output_dir = '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/'
    process_1619_outcomes(input_dir, output_dir)
    #hahahah