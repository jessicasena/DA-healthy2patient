import sys

sys.path.append("/home/jsenadesouza/DA-healthy2patient/code/")

import math
import random
import numpy as np
import logging

from models.util import load_data, set_logger, split_data, get_loaders
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def magnitude(sample):
    mag_vector = []
    for s in sample:
        mag_vector.append(math.sqrt(sum([s[0] ** 2, s[1] ** 2, s[2] ** 2])))
    return mag_vector


def normalize(data):
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data


def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":

    #data_path = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/t900_INTELLIGENT_PAIN_ADAPT_15min.npz"
    data_path = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/PAIN_15min.npz"

    X, y, y_target = load_data(data_path, clin_variable_target="pain_score_class")
    
    labels2idx = {k: idx for idx, k in enumerate(np.unique(y_target))}

    patients = list(np.unique(y[:, -1]))
    random.shuffle(patients)
    patient_splits = np.array_split(patients, 6)
    num_folders = 5
    n_classes = 2
    cum_acc, cum_recall, cum_precision, cum_auc, cum_f1 = [], [], [], [], []
    cum_recall_macro, cum_precision_macro, cum_f1_macro = [], [], []
    for folder_idx in range(num_folders):
        train_data, train_labels, test_data, test_labels, val_data, val_labels = split_data(X, y, y_target, labels2idx,
                                                                                            logging, patient_splits,
                                                                                            folder_idx)

        # nromalize
        train_data = normalize(train_data)
        test_data = normalize(test_data)
        val_data = normalize(val_data)

        #shuffle
        idx = np.random.permutation(len(train_data))
        train_data = train_data[idx]
        train_labels = train_labels[idx]

        model = make_model(input_shape=train_data.shape[1:], num_classes=n_classes)
        #keras.utils.plot_model(model, show_shapes=True)

        epochs = 500
        batch_size = 32

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.h5", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        history = model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2,
            verbose=1,
        )

        metric = "sparse_categorical_accuracy"
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title("model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.show()
        plt.close()