import datetime
import os
import sys
from time import time

import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.utils import class_weight

sys.path.append("/home/jsenadesouza/DA-healthy2patient/code/")

import math
import random
import torch
import numpy as np
import json
import logging
from models.IMU_Transformer.util import utils
from os.path import join
from models.IMU_Transformer.trans_models.IMUTransformer import IMUTransformerEncoder
from models.IMU_Transformer.trans_models.IMUCLSBaseline import IMUCLSBaseline
from models.IMU_Transformer.trans_models.IMU_LSTM import IMU_LSTM
from utils.util import get_metrics, print_metrics, start_timer, end_timer_and_print, validation
from models.util import load_data, set_logger, split_data, get_loaders, load_data_mag, Find_Optimal_Cutoff, get_class_distribution
import torchtest as tt
import io
import msoffcrypto
import glob


def magnitude(sample):
    mag_vector = []
    for s in sample:
        mag_vector.append(math.sqrt(sum([s[0] ** 2, s[1] ** 2, s[2] ** 2])))
    return mag_vector


def test_network(model, loss, optim, batch, device):
    # run all tests
    print(tt.test_suite(
        model,
        loss,  # loss function
        optim,  # optimizer
        batch,  # random data
        test_output_range=False,
        test_vars_change=True,
        test_nan_vals=True,
        test_inf_vals=True,
        test_gpu_available=True,
        device=device
    ))


def train(model, config, device, train_loader, train_labels, use_cuda, val_loader=None):

    start_timer()
    # Get training details
    n_freq_print = config.get("n_freq_print")
    n_epochs = config.get("n_epochs")
    # Load the dataset
    # Set to train mode
    # Load the checkpoint if needed

    logging.info("Initializing from scratch")


    # Set the loss
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights_hugo = loss_weights(train_labels, device)
    loss = torch.nn.NLLLoss(weight=class_weights_hugo)

    # Set the optimizer and scheduler
    optim = torch.optim.Adam(model.parameters(),
                             lr=config.get('lr'),
                             eps=config.get('eps'),
                             weight_decay=config.get('weight_decay'))
    scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                step_size=config.get('lr_scheduler_step_size'),
                                                gamma=config.get('lr_scheduler_gamma'))

    # Train
    checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())
    logging.info("Start training")
    best_loss = 1000000
    losses = []
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(n_epochs):
        model.train(True)
        start = time()
        loss_vals = []
        for minibatch, label in train_loader:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                minibatch["acc"] = minibatch["acc"].to(device)
                minibatch["clin"] = minibatch["clin"].to(device)
                label = label.to(device)
                # Zero the gradients
                optim.zero_grad()

                # Forward pass
                res = model(minibatch)

                # Compute loss
                criterion = loss(res, label)

                # Collect for recoding and plotting
                batch_loss = criterion.item()
                loss_vals.append(batch_loss)

            # Back prop
            scaler.scale(criterion).backward()
            scaler.step(optim)
            # Updates the scale for next iteration.
            scaler.update()
            optim.zero_grad()
        losses.append(np.mean(loss_vals))
        # Scheduler update
        scheduler.step()
        if epoch % n_freq_print == 0:
            # Plot the loss function
            loss_fig_path = checkpoint_prefix + "_loss_fig.png"
            utils.plot_loss_func(np.arange(0, epoch + 1), losses, loss_fig_path)

        # Record loss on train set
        # logging.info("[Epoch-{}] loss: {:.3f}".format(epoch + 1, batch_loss))
        model.train(False)
        if val_loader is not None:
            current_loss, current_metric = validation(model, val_loader, device, loss,
                                                      use_cuda=use_cuda)
        end = time()
        if val_loader is not None:
            epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(n_epochs))).format(epoch + 1)
            epoch_fin = 'train: {:.6f}, val: {:.6f}, F1: {:.4f} [{}]'.format(np.mean(loss_vals), current_loss,
                                                                             current_metric['f1-score_macro'],
                                                                             str(datetime.datetime.timedelta(
                                                                                 seconds=(end - start))))
            logging.info(epoch_desc + epoch_fin)
            if best_loss > current_loss:
                best_loss = current_loss
                torch.save(model.state_dict(), checkpoint_prefix + "_best.pth")
                logging.info("Best model saved")
        else:
            epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(n_epochs))).format(epoch + 1)
            epoch_fin = 'train: {:.6f} [{}]'.format(np.mean(loss_vals), str(datetime.timedelta(seconds=(end - start))))
            logging.info(epoch_desc + epoch_fin)
            # Save checkpoint
            if best_loss > np.mean(loss_vals):
                best_loss = np.mean(loss_vals)
                torch.save(model.state_dict(), checkpoint_prefix + "_best.pth")
                logging.info("Best model saved")

    end_timer_and_print(f"Training session ({n_epochs}epochs)")

    logging.info('Training completed')
    torch.save(model.state_dict(), checkpoint_prefix + '_final.pth')


def plot_accel(sample, title, path, idx):
    import matplotlib.pyplot as plt

    fig = plt.figure(idx)
    plt.plot(sample[0, :])
    plt.plot(sample[1, :])
    plt.plot(sample[2, :])
    plt.title(title)

    plt.savefig(os.path.join(path,f"{title}_{time()}.png"))
    plt.close(fig)


def test(config, device, device_id, test_loader, folder_idx, n_classes):
    checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())
    if config.get("use_model") == "cnn":
        model = IMUCLSBaseline(config).to(device)
    elif config.get("use_model") == "transformer":
        model = IMUTransformerEncoder(config).to(device)
    elif config.get("use_model") == "lstm":
        model = IMU_LSTM(config).to(device)
    else:
        raise ValueError("Model not supported")

    if config.get("checkpoint_path") != "None":
        model.load_state_dict(torch.load(checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(checkpoint_path))
    else:
        # Load the best model
        if config.get("best_model"):
            model.load_state_dict(torch.load(checkpoint_prefix + "_best.pth", map_location=device_id))
        else:
            model.load_state_dict(torch.load(checkpoint_prefix + "_final.pth", map_location=device_id))
    # Set to eval mode
    model.eval()

    logging.info("Start testing")
    predicted = []
    ground_truth = []
    # pred_score = []
    # samples = []
    with torch.no_grad():
        for minibatch, label in test_loader:
            # Forward pass
            minibatch["acc"] = minibatch["acc"].to(device)
            minibatch["clin"] = minibatch["clin"].to(device)
            label = label.to(device)
            res = model(minibatch)

            # Evaluate and append
            pred_label = torch.argmax(res, dim=1)
            #pred_score.append(torch.exp(res).cpu().numpy()[0])
            predicted.extend(pred_label.cpu().numpy())
            ground_truth.extend(label.cpu().numpy())
            #samples.extend(minibatch.cpu().numpy())

    #pred_score = np.array(pred_score)

    #threshold = Find_Optimal_Cutoff(test_labels, pred_score[:, 1])
    #print(f'threshold: {threshold}')
    #y_pred_2 = list(map(lambda x: 1 if x > threshold else 0, pred_score[:, 1]))
    #print(confusion_matrix(ground_truth, y_pred_2))
    # for idx, (sample, true, pred) in enumerate(zip(samples, ground_truth, y_pred_2)):
    #     if true != pred:
    #         plot_accel(sample, f"True: {true} - Pred {pred}", "/home/jsenadesouza/DA-healthy2patient/results/outcomes/missclassified/", idx)
    #path_out = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/IMUTransformers/missclassified/"
    #np.savez_compressed(join(path_out, f"predtest_{folder_idx}.npz"), predicted=y_pred_2, ground_truth=ground_truth, samples=samples)
    def plot_metrics(ground_truth, predicted):
        metrics = get_metrics(ground_truth, predicted)
        for k, v in metrics.items():
            if "confusion" in k:
                logging.info('Fold {} {}:\n{}\n'.format(folder_idx, k.capitalize(), v))
            else:
                logging.info('Fold {} {}: {}\n'.format(folder_idx, k.capitalize(), v))
        return metrics

    metrics = plot_metrics(ground_truth, predicted)
    #metrics = plot_metrics(ground_truth, y_pred_2)

    return metrics


def loss_weights(msk, device):
    weights = [int(sum(msk == 0)),
               int(sum(msk == 1))]
    if weights[0] == 0 or weights[1] == 0:
        weights = torch.FloatTensor([1.0, 1.0]).to(device)
    else:
        if weights[0] > weights[1]:
            weights = torch.FloatTensor([1.0, weights[0] / weights[1]]).to(device)
        else:
            weights = torch.FloatTensor([weights[1] / weights[0], 1.0]).to(device)

    return weights

def set_patient_map():
    # create a map between the subject_deiden_id and the patient id
    patient_map = {}
    patient_enrollment = pd.read_excel('/data/daily_data/patient_id_mapping.xlsx', engine='openpyxl')

    for row in patient_enrollment.itertuples():
        patient_map[row.patient_id] = row.subject_deiden_id

    return patient_map

def get_demo_data(X, y, y_target, y_col_names):
    passwd = 'pervasiveICU'

    filename = '/home/jsenadesouza/DA-healthy2patient/Pervasive_Sensing_Enrollment_Log.xlsx'
    decrypted_workbook = io.BytesIO()
    with open(filename, 'rb') as file:
        office_file = msoffcrypto.OfficeFile(file)
        office_file.load_key(password=passwd)
        office_file.decrypt(decrypted_workbook)

    ADAPT_enrollment = pd.read_excel(decrypted_workbook, engine="openpyxl")

    input_dir = '/data/datasets/ICU_Data/EHR_Data/truncated/2020-02-26/'
    df_2016 = pd.read_csv(os.path.join(input_dir, 'encounters_0_trimmed.csv'))

    df_2021 = []
    files_enc = glob.glob('/data/daily_data/*/encounters*.csv',
                          recursive=True)
    files_peso = glob.glob('/data/daily_data/*/height_weight*.csv',
                           recursive=True)

    for file in files_enc:
        df = pd.read_csv(file)
        df_2021.append(df)

    df_2021_peso = []
    for file in files_peso:
        df = pd.read_csv(file)
        df_2021_peso.append(df)

    df_2021 = pd.concat(df_2021)
    df_2021_peso = pd.concat(df_2021_peso)

    patients_char = []
    patient_map = set_patient_map()
    for patient_id in y_target:
        if "P" in patient_id or "I" in patient_id:
            row = df_2021[df_2021['patient_deiden_id'] == patient_map[patient_id]]
            height = np.mean(df_2021_peso[(df_2021_peso['patient_deiden_id'] == patient_map[patient_id]) & (
                        df_2021_peso['measurement_name'] == 'weight_kgs')]['measurement_value'].values)
            weight = df_2021_peso[(df_2021_peso['patient_deiden_id'] == patient_map[patient_id]) & (
                        df_2021_peso['measurement_name'] == 'height_cm')]['measurement_value'].values[0]
            if "I" in patient_id:
                try:
                    admit = datetime.datetime.strptime(
                        ADAPT_enrollment['ICU_admit'][ADAPT_enrollment["Record ID"] == patient_id].values[0],
                        '%m/%d/%y %H%M')
                except:
                    try:
                        admit = datetime.datetime.strptime(
                            ADAPT_enrollment['ICU_admit'][ADAPT_enrollment["Record ID"] == patient_id].values[0],
                            '%m/%d/%Y %H%M')
                    except:
                        print(
                            f"admit: {ADAPT_enrollment['ICU_admit'][ADAPT_enrollment['Record ID'] == patient_id].values}")
                try:
                    consent = pd.Timestamp(
                        ADAPT_enrollment['Consent Date'][ADAPT_enrollment["Record ID"] == patient_id].values[0])
                except:
                    print(
                        f"consent: {ADAPT_enrollment['Consent Date'][ADAPT_enrollment['Record ID'] == patient_id].values}")
                try:
                    dischg = datetime.datetime.strptime(
                        ADAPT_enrollment['ICU_dischg'][ADAPT_enrollment["Record ID"] == patient_id].values[0],
                        '%m/%d/%y %H%M')
                except:
                    d_time = ADAPT_enrollment['ICU_dischg'][ADAPT_enrollment["Record ID"] == patient_id].values[0]
                    if type(d_time) == type(datetime):
                        dischg = d_time
                    else:
                        dischg = datetime.datetime.combine(datetime.datetime.date.today(), datetime.datetime.min.time())
            else:
                ad_rows = row['admit_datetime'][~row['admit_datetime'].isna()].values
                dc_rows = row['dischg_datetime'][~row['dischg_datetime'].isna()].values
                try:
                    admit = datetime.datetime.strptime(min(ad_rows), '%Y-%m-%d %H:%M:%S')
                except:
                    admit = datetime.datetime.strptime(min(ad_rows), '%Y-%m-%d')
                try:
                    dischg = datetime.datetime.strptime(max(dc_rows), '%Y-%m-%d %H:%M:%S')
                except:
                    try:
                        dischg = datetime.datetime.strptime(max(dc_rows), '%Y-%m-%d')
                    except:
                        print(patient_id)
                        print(dc_rows)
                        dischg = datetime.datetime.combine(datetime.datetime.date.today(), datetime.datetime.min.time())
            consent = admit
        else:
            row = df_2016[df_2016['record_id'] == int(patient_id)]
            height = row['height_cm'][~row['height_cm'].isna()].values[0]
            weight = row['weight_kgs'][~row['weight_kgs'].isna()].values[0]
            ad_rows = row['admit_datetime'][~row['admit_datetime'].isna()].values
            dc_rows = row['dischg_datetime'][~row['dischg_datetime'].isna()].values
            try:
                admit = datetime.datetime.strptime(min(ad_rows), '%Y-%m-%d %H:%M:%S')
            except:
                admit = datetime.datetime.strptime(min(ad_rows), '%Y-%m-%d')
            try:
                dischg = datetime.datetime.strptime(max(dc_rows), '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    dischg = datetime.datetime.strptime(max(dc_rows), '%Y-%m-%d')
                except:
                    print(patient_id)
                    print(dc_rows)
                    dischg = datetime.datetime.combine(datetime.datetime.date.today(), datetime.datetime.min.time())
            consent = admit

        birth = datetime.datetime.strptime(row['birth_date'][~row['birth_date'].isna()].values[0], '%Y-%m-%d')
        age = int((consent - birth).days / 365)
        lenght_stay = abs((dischg - admit).days)

        gender = row['sex'][~row['sex'].isna()].values[0]
        race = row['race'][~row['race'].isna()].values[0]
        ethnicity = row['ethnicity'][~row['ethnicity'].isna()].values[0]
        if len(row['aids'][~row['aids'].isna()]) > 0:
            aids = row['aids'][~row['aids'].isna()].values[0]
        else:
            aids = -1
        if len(row['cancer'][~row['cancer'].isna()]) > 0:
            cancer = row['cancer'][~row['cancer'].isna()].values[0]
        else:
            cancer = -1
        if len(row['cerebrovascular_disease'][~row['cerebrovascular_disease'].isna()]) > 0:
            cerebrovascular_disease = row['cerebrovascular_disease'][~row['cerebrovascular_disease'].isna()].values[0]
        else:
            cerebrovascular_disease = -1
        if len(row['dementia'][~row['dementia'].isna()]) > 0:
            dementia = row['dementia'][~row['dementia'].isna()].values[0]
        else:
            dementia = -1
        if len(row['paraplegia_hemiplegia'][~row['paraplegia_hemiplegia'].isna()]) > 0:
            paraplegia_hemiplegia = row['paraplegia_hemiplegia'][~row['paraplegia_hemiplegia'].isna()].values[0]
        else:
            paraplegia_hemiplegia = -1
        if len(row['smoking_status'][~row['smoking_status'].isna()]) > 0:
            smoking_status = row['smoking_status'][~row['smoking_status'].isna()].values[0]
        else:
            smoking_status = -1
        if len(row['chf'][~row['chf'].isna()]) > 0:
            chf = row['chf'][~row['chf'].isna()].values[0]
        else:
            chf = -1
        if len(row['copd'][~row['copd'].isna()]) > 0:
            copd = row['copd'][~row['copd'].isna()].values[0]
        else:
            copd = -1

        if len(row['diabetes_w_o_complications'][~row['diabetes_w_o_complications'].isna()]) > 0:
            diabetes_w_o_complications = \
            row['diabetes_w_o_complications'][~row['diabetes_w_o_complications'].isna()].values[0]
        else:
            diabetes_w_o_complications = -1
        if len(row['diabetes_w_complications'][~row['diabetes_w_complications'].isna()]) > 0:
            diabetes_w_complications = row['diabetes_w_complications'][~row['diabetes_w_complications'].isna()].values[
                0]
        else:
            diabetes_w_complications = -1
        if diabetes_w_o_complications == 1 or diabetes_w_complications == 1:
            diabetes = 1
        elif diabetes_w_o_complications == -1 and diabetes_w_complications == -1:
            diabetes = -1
        elif diabetes_w_o_complications == 0 and diabetes_w_complications == 0:
            diabetes = 0
        if len(row['m_i'][~row['m_i'].isna()]) > 0:
            m_i = row['m_i'][~row['m_i'].isna()].values[0]
        else:
            m_i = -1
        if len(row['metastatic_carcinoma'][~row['metastatic_carcinoma'].isna()]) > 0:
            metastatic_carcinoma = row['metastatic_carcinoma'][~row['metastatic_carcinoma'].isna()].values[0]
        else:
            metastatic_carcinoma = -1
        if len(row['mild_liver_disease'][~row['mild_liver_disease'].isna()]) > 0:
            mild_liver_disease = row['mild_liver_disease'][~row['mild_liver_disease'].isna()].values[0]
        else:
            mild_liver_disease = -1
        if len(row['moderate_severe_liver_disease'][~row['moderate_severe_liver_disease'].isna()]) > 0:
            moderate_severe_liver_disease = \
            row['moderate_severe_liver_disease'][~row['moderate_severe_liver_disease'].isna()].values[0]
        else:
            moderate_severe_liver_disease = -1
        if mild_liver_disease == 1 or moderate_severe_liver_disease == 1:
            liver_disease = 1
        elif mild_liver_disease == -1 and moderate_severe_liver_disease == -1:
            liver_disease = -1
        elif mild_liver_disease == 0 and moderate_severe_liver_disease == 0:
            liver_disease = 0
        if len(row['peptic_ulcer_disease'][~row['peptic_ulcer_disease'].isna()]) > 0:
            peptic_ulcer_disease = row['peptic_ulcer_disease'][~row['peptic_ulcer_disease'].isna()].values[0]
        else:
            peptic_ulcer_disease = -1
        if len(row['peripheral_vascular_disease'][~row['peripheral_vascular_disease'].isna()]) > 0:
            peripheral_vascular_disease = \
            row['peripheral_vascular_disease'][~row['peripheral_vascular_disease'].isna()].values[0]
        else:
            peripheral_vascular_disease = -1
        if len(row['renal_disease'][~row['renal_disease'].isna()]) > 0:
            renal_disease = row['renal_disease'][~row['renal_disease'].isna()].values[0]
        else:
            renal_disease = -1
        if len(row['rheumatologic_disease'][~row['rheumatologic_disease'].isna()]) > 0:
            rheumatologic_disease = row['rheumatologic_disease'][~row['rheumatologic_disease'].isna()].values[0]
        else:
            rheumatologic_disease = -1

        patients_char.append({'patient_id': patient_id, 'sex': gender, 'race': race, 'height_cm': height,
                              'age': age, 'weight_kgs': weight, 'lenght_stay': lenght_stay,
                              "ethnicity": ethnicity, "aids": aids, "cancer": cancer,
                              "cerebrovascular_disease": cerebrovascular_disease,
                              "dementia": dementia, "paraplegia_hemiplegia": paraplegia_hemiplegia,
                              "smoking_status": smoking_status,
                              "chf": chf, "copd": copd, "diabetes": diabetes, "m_i": m_i,
                              "metastatic_carcinoma": metastatic_carcinoma,
                              "liver_disease": liver_disease, "peptic_ulcer_disease": peptic_ulcer_disease,
                              "renal_disease": renal_disease,
                              "rheumatologic_disease": rheumatologic_disease
                              })

    df_char = pd.DataFrame(data=patients_char)

    df_char.loc[df_char.sex == 'MALE', 'sex'] = 0
    df_char.loc[df_char.sex != 'MALE', 'sex'] = 1
    df_char.loc[df_char.race == 'BLACK', 'race'] = 0
    df_char.loc[df_char.race != 'BLACK', 'race'] = 1
    df_char.loc[df_char.ethnicity == 'HISPANIC', 'ethnicity'] = 0
    df_char.loc[df_char.ethnicity != 'HISPANIC', 'ethnicity'] = 1
    df_char.loc[df_char.smoking_status == 'Former Smoker', 'smoking_status'] = 0
    df_char.loc[df_char.smoking_status == 'Smoker', 'smoking_status'] = 1
    df_char.loc[df_char.smoking_status == 'Smoker, Current Status Unknown', 'smoking_status'] = 1
    df_char.loc[df_char.smoking_status == 'Current Every Day Smoker', 'smoking_status'] = 1
    df_char.loc[df_char.smoking_status == 'Current Some Day Smoker', 'smoking_status'] = 1
    df_char.loc[df_char.smoking_status == 'Light Tobacco Smoker', 'smoking_status'] = 1
    df_char.loc[df_char.smoking_status == 'Never Smoker', 'smoking_status'] = 2
    df_char.loc[df_char.smoking_status == 'Never Smoker ', 'smoking_status'] = 2
    df_char.loc[df_char.smoking_status == 'Status Unknown', 'smoking_status'] = 3
    df_char.loc[df_char.smoking_status == 'Unknown If Ever Smoked', 'smoking_status'] = 3
    df_char.loc[df_char.smoking_status == 'Current Status Unknown', 'smoking_status'] = 3
    df_char.loc[df_char.smoking_status == 'Unknown If Ever Smoked', 'smoking_status'] = 3
    df_char.loc[df_char.smoking_status == 'Never Assessed', 'smoking_status'] = 3

    X_char = []
    col_patient = y_col_names.index('patient_id')
    for xx, sample in zip(X.squeeze(), y):
        try:
            char_pat = df_char[df_char["patient_id"] == sample[col_patient]]
            char_final = list(char_pat.loc[:, char_pat.columns != "patient_id"].values[0])
            X_char.append(char_final)
        except:
            print(sample[col_patient])
    X_char = np.array(X_char)
    return X_char, df_char


def split(y):
    folds_pat = []
    folds_pat.append(
        ['P023', 'P013', 'I051A', '49', '48', '100', 'I045A', 'P051', 'I028A', '112', '92', '83', 'P037', '22', '29',
         '35', '64', '58', 'P046', 'I001A', 'I034A'])
    folds_pat.append(
        ['I021A', 'I043A', 'I019A', 'I044A', 'I008A', '51', '106', 'P004', 'P007', '82', '69', 'P054', 'P055', '63',
         '41', '4', '89', 'I027A', 'P024', '17'])
    folds_pat.append(
        ['P038', 'P021', 'P017', 'P067', 'I033A', 'I050A', 'P052', '40', 'P015', '109', '90', 'I049A', '103', '14',
         '81', '60', '20', 'I047A', 'P006', '32'])
    folds_pat.append(
        ['I052A', '98', 'P029', 'I006A', 'I037A', 'P057', 'I004A', 'P042', 'I018A', 'P028', '25', 'P003', 'I025A', '39',
         '52', '28', '18', 'I053A', 'I023A', '8'])
    folds_pat.append(
        ['P010', 'I026A', '95', 'I042A', 'P063', '88', '66', '50', '93', '87', '65', '44', 'I022A', 'P070', '47', '26',
         '75', 'P009', '13', '15'])

    folds_idx = [[], [], [], [], []]
    for i in range(5):
        for pat in folds_pat[i]:
            idxs = np.where(y[:, -1] == pat)[0]
            folds_idx[i].extend(list(idxs))

            # %%

    folders = [[[], []], [[], []], [[], []], [[], []], [[], []]]
    for i in range(len(folds_idx)):
        # print(f"Folder {i}")
        for j in range(len(folds_idx)):
            if i == j:
                # print(f"Train:{j}")
                folders[i][0].append(folds_idx[j])
            else:
                # print(f"Test:{j}")
                folders[i][1].extend(folds_idx[j])
    return folders


if __name__ == "__main__":
    # Read configuration
    with open('/home/jsenadesouza/DA-healthy2patient/code/models/IMU_Transformer/config.json', "r") as read_file:
        config = json.load(read_file)

    # Set the seeds and the device
    torch_seed = 42
    numpy_seed = 42
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
    random.seed(42)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    else:
        device_id = 'cpu'

    device = torch.device(device_id)

    num_folders = 5
    n_classes = 2

    utils.init_logger()
    # Record execution details
    logging.info("Starting IMU-transformers")
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    if config.get("use_model") == "cnn":
        model = IMUCLSBaseline(config).to(device)
    elif config.get("use_model") == "transformer":
        model = IMUTransformerEncoder(config).to(device)
    elif config.get("use_model") == "lstm":
        model = IMU_LSTM(config).to(device)
    else:
        raise ValueError("Model not supported")

    # Set the dataset and data loader
    logging.info("Start train data preparation")
    # read data and get dataset and dataloader
    # split the data into train, val and test sets
    X, y, y_target, y_col_names = load_data(config.get("data_path"), clin_variable_target="pain_score_class")
    labels2idx = {k: idx for idx, k in enumerate(np.unique(y_target))}
    yy_t = np.array([0 if x == "mild" else 1 for x in y_target])


    col_idx_prevpain = y_col_names.index('pain_score_prev_class')
    prev_pain = y[:, col_idx_prevpain]
    prev_pain = np.array([0 if x == "mild" else 1 for x in prev_pain])

    #X_demo, df_demo = get_demo_data(X, y, y[:, -1], y_col_names)

    folders = split(y)

    sample_start = X.shape[1] - config.get("sample_size")
    checkpoint_path = config.get("checkpoint_path")

    cum_acc, cum_recall, cum_precision, cum_auc, cum_f1 = [], [], [], [], []
    cum_recall_macro, cum_precision_macro, cum_f1_macro = [], [], []

    for folder_idx in range(num_folders):
        clin_data = prev_pain

        train_idx = folders[folder_idx][0]
        test_idx = folders[folder_idx][1]
        train_acc_data, train_labels, test_acc_data, test_labels = X[train_idx], yy_t[train_idx], X[test_idx], yy_t[test_idx]
        train_clin_data, test_clin_data = clin_data[train_idx], clin_data[test_idx]
        train_data = {"acc": train_acc_data, "clin": train_clin_data}
        test_data = {"acc": test_acc_data, "clin": test_clin_data}


        logging.info(f"Folder {folder_idx + 1}")
        logging.info(f"Train data: {get_class_distribution(np.unique(train_labels, return_counts=True))}")
        logging.info(f"Test data: {get_class_distribution(np.unique(test_labels, return_counts=True))}")

        # get dataloaders
        train_loader, test_loader = get_loaders(config.get("batch_size"), sample_start, train_data, train_labels, test_data, test_labels)


        logging.info("Train data shape: {}".format(train_data["acc"].shape))
        logging.info("Train data shape: {}".format(train_data["clin"].shape))
        # zero = np.where(train_labels == 0)[0][0]
        # one = np.where(train_labels == 1)[0][0]
        # train_data, train_labels = train_data[[zero, one]], train_labels[[zero, one]]
        # train_loader, test_loader, val_loader = get_loaders(2, train_data, train_labels,
        #                                                     test_data, test_labels,
        #                                                     val_data, val_labels, weighted_sampler=False)
        if config.get("checkpoint_path") == "None":
            train(model, config, device, train_loader, train_labels, use_cuda)
        metrics = test(config, device, device_id, test_loader, folder_idx, n_classes)

        logging.info("Data preparation completed")
        cum_acc.append(metrics['accuracy'])
        cum_f1.append(metrics['f1-score'])
        cum_recall.append(metrics['recall'])
        cum_precision.append(metrics['precision'])
        cum_auc.append(metrics['roc_auc'])
        cum_f1_macro.append(metrics['f1-score_macro'])
        cum_recall_macro.append(metrics['recall_macro'])
        cum_precision_macro.append(metrics['precision_macro'])

    print_metrics(logging, n_classes, cum_acc, cum_recall, cum_precision, cum_auc, cum_f1, cum_recall_macro,
                  cum_precision_macro,
                  cum_f1_macro)






    # Record overall statistics
    # confusion_mat = confusion_matrix(ground_truth, predicted, labels=list(range(config.get("num_classes"))))
    # print(confusion_mat.shape)
    # stats_msg = "\n\tAccuracy: {:.3f}".format(np.mean(metric))
    # accuracies = []
    # for i in range(len(accuracy_per_label)):
    #     print("Performance for class [{}] - accuracy {:.3f}".format(i, accuracy_per_label[i] / count_per_label[i]))
    #     accuracies.append(accuracy_per_label[i] / count_per_label[i])
    # # save dump
    # np.savez(checkpoint_path + "_test_results_dump", confusion_mat=confusion_mat, accuracies=accuracies,
    #          count_per_label=count_per_label, total_acc=np.mean(metric))
    # logging.info(stats_msg)
