import numpy as np
from scipy import stats as st

file = open("/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/exp_transformer_pain_score_class_1663180892.865274_focalloss.log")
cum_acc, cum_f1, cum_recall, cum_precision, cum_auc = [], [], [], [], []
for line in file:
    if "Accuracy" in line:
        cum_acc.append(float(line.split("Accuracy: ")[1].split("\n")[0]))
    elif "F1" in line:
        cum_f1.append([float(line.split("F1-score: [")[1].split(" ")[0]), float(line.split("F1-score: [")[1].split(" ")[1].split("]")[0])])
    elif "Recall" in line:
        cum_recall.append([float(line.split("Recall: [")[1].split(" ")[0]), float(line.split("Recall: [")[1].split(" ")[1].split("]")[0])])
    elif "Precision" in line:
        cum_precision.append([float(line.split("Precision: [")[1].split(" ")[0]), float(line.split("Precision: [")[1].split(" ")[1].split("]")[0])])
    elif "Roc_auc" in line:
        cum_auc.append(float(line.split("Roc_auc: ")[1].split("\n")[0]))

for class_ in range(2):
    print(f"Class: {class_}")
    current_acc = np.array(cum_acc)
    current_f1 = np.array(cum_f1)[:, class_]
    current_recall = np.array(cum_recall)[:, class_]
    current_prec = np.array(cum_precision)[:, class_]
    current_auc = np.array(cum_auc)
    ci_mean = st.t.interval(0.95, len(current_acc) - 1, loc=np.mean(current_acc), scale=st.sem(current_acc))
    ci_f1 = st.t.interval(0.95, len(current_f1) -1, loc=np.mean(current_f1), scale=st.sem(current_f1))
    ci_recall = st.t.interval(0.95, len(current_recall) -1, loc=np.mean(current_recall), scale=st.sem(current_recall))
    #ci_auc = st.t.interval(0.95, len(cum_auc) -1, loc=np.mean(cum_auc), scale=st.sem(cum_auc))
    # ci_AUROC = st.t.interval(0.95, len(cum_AUROC) -1, loc=np.mean(cum_AUROC), scale=st.sem(cum_AUROC))
    ci_prec = st.t.interval(0.95, len(current_prec) -1, loc=np.mean(current_prec), scale=st.sem(current_prec))
    ci_auc = st.t.interval(0.95, len(current_auc) -1, loc=np.mean(current_auc), scale=st.sem(current_auc))

    print('accuracy: {:.2f} ± {:.2f}\n'.format(np.mean(current_acc) * 100, abs(np.mean(current_acc) - ci_mean[0]) * 100))
    print('recall: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall) * 100, abs(np.mean(current_recall) - ci_recall[0]) * 100))
    print('f1-score: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1) * 100, abs(np.mean(current_f1) - ci_f1[0]) * 100))
    print('precision: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec) * 100, abs(np.mean(current_prec) - ci_prec[0]) * 100))
    print('roc_auc: {:.2f} ± {:.2f}\n'.format(np.mean(current_auc) * 100, abs(np.mean(current_auc) - ci_auc[0]) * 100))