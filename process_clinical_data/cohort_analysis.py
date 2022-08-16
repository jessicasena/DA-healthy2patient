import os
dir_dataset = "/data/datasets/ICU_Data/Sensor_Data/"
for root, dirs, files in os.walk(dir_dataset):
    for file in files:
        if file.endswith("RAW.csv"):
            patient = root.split('/')[5].split("_")[1]
            timestamp = file.split("(")[-1].split(")")[0]
            bodypart = file.split("_")[0]
            acc_csv = os.path.join(root, file)
            # just get csv files from Accelerometer directories
            # if os.path.join(dir_dataset, f"Patient_{patient}", "Accelerometer") in root:
            #     if acc_csv not in accs:
            #         accs[acc_csv] = {'patient': patient, 'timestamp': timestamp, "bodypart": bodypart}
            # else:
            #     no_acc += 1