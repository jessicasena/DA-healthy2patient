import numpy as np


data_input_file = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/f10_t900_IntelligentICU_PAIN_ADAPT_15min.npz"
tmp = np.load(data_input_file, allow_pickle=True)
X= tmp['X']
y = tmp['y']
y_col_names = list(tmp['y_col_names'])
col_idx = y_col_names.index('patient_id')
y_patient = np.array(y[:, col_idx])
y_target = np.unique(y_patient)
ok =0
