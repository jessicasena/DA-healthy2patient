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
import glob
import numpy as np
import sys


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


def process_vitals(outcomes, patient_map):

    files = glob.glob('/data/daily_data/*/vitals*.csv',
                      recursive=True)
    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            # standardize the timestamp
            timestamp = datetime.strptime(row['vitals_datetime'], '%Y-%m-%d %H:%M:%S')
            timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
            patient_id = patient_map[row.patient_deiden_id]
            key = f"{timestamp}_{patient_id}"
            if row.measurement_name == "heart_rate":
                if not pd.isna(row.measurement_value):
                    hr = float(row.measurement_value)
                else:
                    hr = -1
                if key in outcomes:
                    outcomes[key]['heart_rate'] = hr
                else:
                    outcomes[key] = {'heart_rate': hr}
            elif row.measurement_name == "temp_farenheit":
                if not pd.isna(row.measurement_value):
                    temp = float(row.measurement_value)
                else:
                    temp = -1
                if key in outcomes:
                    outcomes[key]['temp_farenheit'] = temp
                else:
                    outcomes[key] = {'temp_farenheit': temp}

        return outcomes


def process_encounters(outcomes, patient_map):
    files = glob.glob('/data/daily_data/*/encounters*.csv',
                      recursive=True)

    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            try:
                admit = datetime.strptime(row['admit_datetime'], '%Y-%m-%d')
                dischg = datetime.strptime(row['dischg_datetime'], '%Y-%m-%d')
                lenght_stay = abs((dischg - admit).days)
                key = patient_map[row.patient_deiden_id]
                is_dead = 1 if type(row['death_date']) == str else 0

                if key in outcomes:
                    outcomes[key]['lenght_stay'] = lenght_stay
                    outcomes[key]['is_dead'] = is_dead
                else:
                    outcomes[key] = {'lenght_stay': lenght_stay, 'is_dead': is_dead}
            except:
                #print(f"Error with {row['patient_deiden_id']}: {row['admit_datetime']} {row['dischg_datetime']} {row['death_date']}")
                pass
    return outcomes


def process_dvprs(value):
    try:
        score = int(value.split(' ')[0])
    except:
        if value == 'Patient Asleep':
            score = -1
        elif np.isnan(value):
            score = -1
        else:
            sys.exit('error in dvprs')

    return score


def process_painscore(outcomes, patient_map):
    files = glob.glob('/data/daily_data/*/pain*.csv',
                      recursive=True)
    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            try:
                # standardize the timestamp
                # timestamp_patient_id
                timestamp = datetime.strptime(row['pain_datetime'], '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
                patient_id = patient_map[row.patient_deiden_id]
                key = f"{timestamp}_{patient_id}"
                if row.measurement_name == "pain_uf_dvprs":
                    try:
                        score = process_dvprs(row.measurement_value)
                    except:
                        score = -1

                    if key in outcomes:
                        outcomes[key]['pain_score'] = score
                    else:
                        outcomes[key] = {'pain_score': score}
            except KeyError as e:
                #print(e)
                pass
    return outcomes


def process_sofascore(outcomes, patient_map):
    files = glob.glob('/data/daily_data/*/sofa*.csv',
                      recursive=True)
    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            try:
                # standardize the timestamp
                # timestamp_patient_id
                timestamp = datetime.strptime(row['date_of_care'], '%Y-%m-%d')
                timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
                patient_id = patient_map[row.patient_deiden_id]
                key = f"{timestamp}_{patient_id}"
                score = row.SOFA_SCORE
                if key in outcomes:
                    outcomes[key]['sofa_score'] = score
                else:
                    outcomes[key] = {'sofa_score': score}
            except KeyError as e:
                #print(e)
                pass
    return outcomes


def process_blood_pressure(outcomes, patient_map):
    files = glob.glob('/data/daily_data/*/blood_pressure*.csv',
                      recursive=True)
    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            try:
                timestamp = datetime.strptime(row['bp_datetime'], '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
                patient_id = patient_map[row.patient_deiden_id]
                key = f"{timestamp}_{patient_id}"
                if row.measurement_name == "map_a_line" or row.measurement_name == "map_non_invasive":
                    score = int(row.measurement_value)

                    if key in outcomes:
                        outcomes[key]['blood_pressure'] = score
                    else:
                        outcomes[key] = {'blood_pressure': score}
            except KeyError as e:
                #print(e)
                pass
    return outcomes


def process_braden(outcomes, patient_map):

    files = glob.glob('/data/daily_data/*/braden*.csv',
                      recursive=True)

    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            try:
                timestamp = datetime.strptime(row['braden_datetime'], '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
                patient_id = patient_map[row.patient_deiden_id]
                key = f"{timestamp}_{patient_id}"
                if row.measurement_name == "braden_mobility":
                    score = int(row.measurement_value)

                    if key in outcomes:
                        outcomes[key]['braden_score'] = score
                    else:
                        outcomes[key] = {'braden_score': score}
            except KeyError as e:
                # print(e)
                pass
    return outcomes


def process_pulse_ox(outcomes, patient_map):

    files = glob.glob('/data/daily_data/*/respiratory*.csv',
                      recursive=True)

    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            try:
                timestamp = datetime.strptime(row['respiratory_datetime'], '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
            except:
                timestamp = datetime.strptime(row['respiratory_datetime'], '%Y-%m-%d')
                timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
            try:
                patient_id = patient_map[row.patient_deiden_id]
                key = f"{timestamp}_{patient_id}"
                if row.measurement_name == "respiratory_rate":
                    score = int(row.measured_value)

                    if key in outcomes:
                        outcomes[key]['spo2'] = score
                    else:
                        outcomes[key] = {'spo2': score}
            except:
                pass

    return outcomes


def process_cam(outcomes, patient_map):
    files = glob.glob('/data/daily_data/*/cam*.csv',
                      recursive=True)

    for file in files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            try:
                timestamp = datetime.strptime(row['recorded_time'], '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
            except:
                timestamp = datetime.strptime(row['recorded_time'], '%Y-%m-%d')
                timestamp = timestamp.strftime('%m-%d-%Y %H:%M')
            try:
                patient_id = patient_map[row.patient_deiden_id]
                key = f"{timestamp}_{patient_id}"
                if row.vital_sign_measure_name == "R PALLIATIVE (CAM) SCORE":
                    score = int(bool(row.meas_value))

                    if key in outcomes:
                        outcomes[key]['cam'] = score
                    else:
                        outcomes[key] = {'cam': score}
            except:
                pass

    return outcomes

def set_patient_map():
    # create a map between the subject_deiden_id and the patient id
    patient_map = {}
    patient_enrollment = pd.read_excel('/data/daily_data/patient_id_mapping.xlsx')

    for row in patient_enrollment.itertuples():
        patient_map[row.subject_deiden_id] = row.patient_id

    return patient_map



def process_2022_outcomes(output_dir):
    """
    Process 1619 outcomes files.
    """
    outcomes = {}
    outcomes_all = {}
    patient_map = set_patient_map()
    outcomes = process_vitals(outcomes, patient_map)
    outcomes = process_painscore(outcomes, patient_map)
    outcomes = process_sofascore(outcomes, patient_map)
    outcomes = process_blood_pressure(outcomes, patient_map)
    outcomes = process_braden(outcomes, patient_map)
    outcomes = process_pulse_ox(outcomes, patient_map)
    outcomes = process_cam(outcomes, patient_map)
    outcomes_all = process_encounters(outcomes_all, patient_map)

    dict_list = []
    for key, value in outcomes.items():
        patient_id = key.split('_')[1]
        timestamp = datetime.strptime(key.split('_')[0], '%m-%d-%Y %H:%M')
        dict = {'timestamp': timestamp, 'patient_id': patient_id}
        dict.update(value)
        if patient_id in outcomes_all:
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

    df.to_csv(os.path.join(output_dir, '2020-2022_clinical_data_outcomes.csv'), index=False)

if __name__ == '__main__':
    output_dir = '/home/jsenadesouza/DA-healthy2patient/data/clinical_data/'
    process_2022_outcomes(output_dir)