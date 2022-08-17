from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st
import os
import random
import numpy as np
import tensorflow as tf

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Lambda, Input
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras import backend as K
from sklearn import metrics
from layers import SelfAttention

def model(x_train, num_labels, LSTM_units, num_conv_filters, batch_size, F, D):
    """
    The proposed model with CNN layer, LSTM RNN layer and self attention layers.
    Inputs:
    - x_train: required for creating input shape for RNN layer in Keras
    - num_labels: number of output classes (int)
    - LSTM_units: number of RNN units (int)
    - num_conv_filters: number of CNN filters (int)
    - batch_size: number of samples to be processed in each batch
    - F: the attention length (int)
    - D: the length of the output (int) 
    Returns
    - model: A Keras model
    """
    cnn_inputs = Input(shape=(x_train.shape[1], x_train.shape[2], 1), batch_size=batch_size, name='rnn_inputs')
    cnn_layer = Conv2D(num_conv_filters, kernel_size = (1, x_train.shape[2]), strides=(1, 1), padding='valid', data_format="channels_last")
    cnn_out = cnn_layer(cnn_inputs)

    sq_layer = Lambda(lambda x: K.squeeze(x, axis = 2))
    sq_layer_out = sq_layer(cnn_out)

    rnn_layer = LSTM(LSTM_units, return_sequences=True, name='lstm', return_state=True) #return_state=True
    rnn_layer_output, _, _ = rnn_layer(sq_layer_out)

    encoder_output, attention_weights = SelfAttention(size=F, num_hops=D, use_penalization=False, batch_size = batch_size)(rnn_layer_output)
    dense_layer = Dense(num_labels, activation = 'softmax')
    dense_layer_output = dense_layer(encoder_output)

    model = Model(inputs=cnn_inputs, outputs=dense_layer_output)
    print (model.summary())

    return model

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


EPOCH = 10
BATCH_SIZE = 16
LSTM_UNITS = 32
CNN_FILTERS = 3
NUM_LSTM_LAYERS = 1
LEARNING_RATE = 1e-4
PATIENCE = 20
SEED = 0
F = 32
D = 10
DATA_FILES = ['WISDM.npz']
MODE = 'LOTO'
BASE_DIR = '/home/jsenadesouza/DA-healthy2patient/code/ConvLSTMSelfAttention/data/' + MODE + '/'
SAVE_DIR = '/home/jsenadesouza/DA-healthy2patient/code/ConvLSTMSelfAttention/' + MODE + '_results'

if not os.path.exists(os.path.join(SAVE_DIR)):
    os.mkdir(os.path.join(SAVE_DIR))

if __name__ == '__main__':
    SEED = 0 
    random.seed(SEED)
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(0)

    for DATA_FILE in DATA_FILES:
        data_input_file = os.path.join(BASE_DIR, DATA_FILE)
        tmp = np.load(data_input_file, allow_pickle=True)
        X = tmp['X']
        X = np.squeeze(X, axis = 1)
        y_one_hot = tmp['y']
        folds = tmp['folds']

        NUM_LABELS = y_one_hot.shape[1]

        avg_acc = []
        avg_recall = []
        avg_f1 = []
        early_stopping_epoch_list = []
        y = np.argmax(y_one_hot, axis=1)

        for i in range(0, len(folds)):
            train_idx = folds[i][0]
            test_idx = folds[i][1]

            X_train, y_train, y_train_one_hot = X[train_idx], y[train_idx], y_one_hot[train_idx]
            X_test, y_test, y_test_one_hot = X[test_idx], y[test_idx], y_one_hot[test_idx]

            X_train_ = np.expand_dims(X_train, axis = 3)
            X_test_ = np.expand_dims(X_test, axis = 3)

            train_trailing_samples =  X_train_.shape[0]%BATCH_SIZE
            test_trailing_samples =  X_test_.shape[0]%BATCH_SIZE


            if train_trailing_samples!= 0:
                X_train_ = X_train_[0:-train_trailing_samples]
                y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
                y_train = y_train[0:-train_trailing_samples]
            if test_trailing_samples!= 0:
                X_test_ = X_test_[0:-test_trailing_samples]
                y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]
                y_test = y_test[0:-test_trailing_samples]

            print (y_train.shape, y_test.shape)   

            rnn_model = model(x_train = X_train_, num_labels = NUM_LABELS, LSTM_units = LSTM_UNITS, \
                num_conv_filters = CNN_FILTERS, batch_size = BATCH_SIZE, F = F, D= D)

            model_filename = SAVE_DIR + '/best_model_with_self_attn_' + str(DATA_FILE[0:-4]) + '_fold_' + str(i) + '.h5'
            callbacks = [ModelCheckpoint(filepath=model_filename, monitor = 'val_accuracy', save_weights_only=True, save_best_only=True), EarlyStopping(monitor='val_acc', patience=PATIENCE)]#, LearningRateScheduler()]

            opt = optimizers.Adam(clipnorm=1.)

            rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            history = rnn_model.fit(X_train_, y_train_one_hot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(X_test_, y_test_one_hot))

            early_stopping_epoch = callbacks[1].stopped_epoch - PATIENCE + 1
            print('Early stopping epoch: ' + str(early_stopping_epoch))
            early_stopping_epoch_list.append(early_stopping_epoch)

            if early_stopping_epoch <= 0:
                early_stopping_epoch = -100

            # Evaluate model and predict data on TEST 
            print("******Evaluating TEST set*********")
            rnn_model.load_weights(model_filename)

            y_test_predict = rnn_model.predict(X_test_, batch_size = BATCH_SIZE)
            y_test_predict = np.array(y_test_predict)
            y_test_predict = np.argmax(y_test_predict, axis=1)

            all_trainable_count = int(np.sum([K.count_params(p) for p in set(rnn_model.trainable_weights)]))

            MAE = metrics.mean_absolute_error(y_test, y_test_predict, sample_weight=None, multioutput='uniform_average')

            acc_fold = accuracy_score(y_test, y_test_predict)
            avg_acc.append(acc_fold)

            recall_fold = recall_score(y_test, y_test_predict, average='macro')
            avg_recall.append(recall_fold)

            f1_fold  = f1_score(y_test, y_test_predict, average='macro')
            avg_f1.append(f1_fold)

            with open(SAVE_DIR + '/results_model_with_self_attn_' + MODE + '.csv', 'a') as out_stream:
                out_stream.write(str(SEED) + ', ' + str(DATA_FILE[0:-4]) + ', ' + str(i) + ', ' + str(early_stopping_epoch) + ', ' + str(all_trainable_count) + ', ' + str(acc_fold) + ', ' + str(MAE) + ', ' + str(recall_fold) + ', ' + str(f1_fold) + '\n')


            print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
            print('______________________________________________________')
            K.clear_session()

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc = np.mean(avg_f1), scale=st.sem(avg_f1))

    print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))
