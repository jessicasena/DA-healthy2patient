import sys
sys.path.append("/home/jsenadesouza/DA-healthy2patient/code/")
from tensorflow.keras.layers import Add, Dense, Dropout, MultiHeadAttention, LayerNormalization, Layer, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback

import math
import wandb
import numpy as np
import tensorflow as tf
from models.util import load_data, split_data
import random


def init_logger(X_train, y_train, X_val, y_val):
    wandb.login()

    sweep_config = {
        'method': 'grid',
        'metric': {
            'goal': 'maximize',
            'name': 'val_accuracy'
        },
        'parameters': {
            'epochs': {
                'value': 1
            },
            'num_layers': {
                'value': 3
            },
            'embed_layer_size': {
                'value': 128
            },
            'fc_layer_size': {
                'value': 256
            },
            'num_heads': {
                'value': 6
            },
            'dropout': {
                'value': 0.1
            },
            'attention_dropout': {
                'value': 0.1
            },
            'optimizer': {
                'value': 'adam'
            },
            'amsgrad': {
                'value': False
            },
            'label_smoothing': {
                'value': 0.1
            },
            'learning_rate': {
                'value': 1e-3
            },
            # 'weight_decay': {
            #    'values': [2.5e-4, 1e-4, 5e-5, 1e-5]
            # },
            'warmup_steps': {
                'value': 10
            },
            'batch_size': {
                'value': 64
            },
            'global_clipnorm': {
                'value': 3.0
            },
            'X_train': {
                'value': X_train
            },
            'y_train': {
                'value': y_train
            },
            'X_val': {
                'value': X_val
            },
            'y_val': {
                'value': y_val
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="imu-transformer")

    return sweep_id


class PositionalEmbedding(Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.units = units

        self.projection = Dense(units, kernel_initializer=TruncatedNormal(stddev=0.02))

        self.dropout = Dropout(rate=dropout_rate)

    def build(self, input_shape):
        super(PositionalEmbedding, self).build(input_shape)

        self.position = self.add_weight(
            name="position",
            shape=(1, input_shape[1], self.units),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs, training):
        x = self.projection(inputs)
        x = x + self.position

        return self.dropout(x, training=training)


class Encoder(Layer):
    def __init__(
        self, embed_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate, **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=attention_dropout_rate,
            kernel_initializer=TruncatedNormal(stddev=0.02),
        )

        self.dense_0 = Dense(
            units=mlp_dim,
            activation="gelu",
            kernel_initializer=TruncatedNormal(stddev=0.02),
        )
        self.dense_1 = Dense(
            units=embed_dim, kernel_initializer=TruncatedNormal(stddev=0.02)
        )

        self.dropout_0 = Dropout(rate=dropout_rate)
        self.dropout_1 = Dropout(rate=dropout_rate)

        self.norm_0 = LayerNormalization(epsilon=1e-5)
        self.norm_1 = LayerNormalization(epsilon=1e-5)

        self.add_0 = Add()
        self.add_1 = Add()

    def call(self, inputs, training):
        # Attention block
        x = self.norm_0(inputs)
        x = self.mha(
            query=x,
            value=x,
            key=x,
            training=training,
        )
        x = self.dropout_0(x, training=training)
        x = self.add_0([x, inputs])

        # MLP block
        y = self.norm_1(x)
        y = self.dense_0(y)
        y = self.dense_1(y)
        y = self.dropout_1(y, training=training)

        return self.add_1([x, y])


class Transformer(Model):
    def __init__(
        self,
        num_layers,
        embed_dim,
        mlp_dim,
        num_heads,
        num_classes,
        dropout_rate,
        attention_dropout_rate,
        **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)

        # Input (normalization of RAW measurements)
        self.input_norm = Normalization()

        # Input
        self.pos_embs = PositionalEmbedding(embed_dim, dropout_rate)

        # Encoder
        self.e_layers = [
            Encoder(embed_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate)
            for _ in range(num_layers)
        ]

        # Output
        self.norm = LayerNormalization(epsilon=1e-5)
        self.final_layer = Dense(num_classes, kernel_initializer="zeros")

    def call(self, inputs, training):
        x = self.input_norm(inputs)
        x = self.pos_embs(x, training=training)

        for layer in self.e_layers:
            x = layer(x, training=training)

        x = self.norm(x)
        x = self.final_layer(x)

        return x


def smoothed_sparse_categorical_crossentropy(label_smoothing: float = 0.0):
    def loss_fn(y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(y_true, num_classes)

        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=label_smoothing)
        return tf.reduce_mean(loss)

    return loss_fn


def cosine_schedule(base_lr, total_steps, warmup_steps):
    def step_fn(epoch):
        lr = base_lr
        epoch += 1

        progress = (epoch - warmup_steps) / float(total_steps - warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)

        lr = lr * 0.5 * (1.0 + tf.cos(math.pi * progress))

        if warmup_steps:
            lr = lr * tf.minimum(1.0, epoch / warmup_steps)

        return lr

    return step_fn


class PrintLR(Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"lr": self.model.optimizer.lr.numpy()}, commit=False)


def train_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Generate new model
        model = Transformer(
            num_layers=config.num_layers,
            embed_dim=config.embed_layer_size,
            mlp_dim=config.fc_layer_size,
            num_heads=config.num_heads,
            num_classes=18,
            dropout_rate=config.dropout,
            attention_dropout_rate=config.attention_dropout,
        )

        # adapt on training dataset - must be before model.compile !!!
        model.input_norm.adapt(config.X_train, batch_size=config.batch_size)
        print(model.input_norm.variables)

        # Select optimizer
        if config.optimizer == "adam":
            optim = Adam(
                global_clipnorm=config.global_clipnorm,
                amsgrad=config.amsgrad,
            )
        elif config.optimizer == "adamw":
            optim = AdamW(
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
                global_clipnorm=config.global_clipnorm,
                exclude_from_weight_decay=["position"]
            )
        else:
            raise ValueError("The used optimizer is not in list of available")

        model.compile(
            loss=smoothed_sparse_categorical_crossentropy(label_smoothing=config.label_smoothing),
            optimizer=optim,
            metrics=["accuracy"],
        )

        # Train model
        model.fit(
            config.X_train,
            config.y_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=(config.X_val, config.y_val),
            callbacks=[
                LearningRateScheduler(cosine_schedule(base_lr=config.learning_rate, total_steps=config.epochs,
                                                      warmup_steps=config.warmup_steps)),
                PrintLR(),
                WandbCallback(monitor="val_accuracy", mode='max', save_weights_only=True),
                EarlyStopping(monitor="val_accuracy", mode='max', min_delta=0.001, patience=5),
            ],
            verbose=1
        )

        model.summary()

def train(X_train, y_train, X_val, y_val):
    num_layers = 3
    embed_layer_size = 128
    fc_layer_size = 256
    num_heads = 6
    num_classes = 2
    dropout = 0.1
    attention_dropout = 0.1
    batch_size = 128
    global_clipnorm = 3.0
    optimizer = "adam"
    amsgrad = False
    weight_decay = 2.5e-4
    epochs = 1
    learning_rate = 1e-3
    warmup_steps = 10
    label_smoothing = 0.1

    # Generate new model
    model = Transformer(
        num_layers=num_layers,
        embed_dim=embed_layer_size,
        mlp_dim=fc_layer_size,
        num_heads=num_heads,
        num_classes=num_classes,
        dropout_rate=dropout,
        attention_dropout_rate=attention_dropout,
    )

    # adapt on training dataset - must be before model.compile !!!
    model.input_norm.adapt(X_train, batch_size=batch_size)
    print(model.input_norm.variables)

    # Select optimizer
    if optimizer == "adam":
        optim = Adam(
            global_clipnorm=global_clipnorm,
            amsgrad=amsgrad,
        )
    elif optimizer == "adamw":
        optim = AdamW(
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            global_clipnorm=global_clipnorm,
            exclude_from_weight_decay=["position"]
        )
    else:
        raise ValueError("The used optimizer is not in list of available")

    model.compile(
        loss=smoothed_sparse_categorical_crossentropy(label_smoothing=label_smoothing),
        optimizer=optim,
        metrics=["accuracy"],
    )

    # Train model
    model.fit(
        X_train,
         y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[
            LearningRateScheduler(cosine_schedule(base_lr=learning_rate, total_steps=epochs,
                                                  warmup_steps=warmup_steps)),
            # PrintLR(),
            # WandbCallback(monitor="val_accuracy", mode='max', save_weights_only=True),
            EarlyStopping(monitor="val_accuracy", mode='max', min_delta=0.001, patience=5),
        ],
        verbose=1
    )

    model.summary()
    return model


def test(X_test, y_test):
    # Generate new model
    model = Transformer(
        num_layers=3,
        embed_dim=128,
        mlp_dim=256,
        num_heads=6,
        num_classes=18,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )
    model.load_weights("./save/model-best.h5")

    def get_predictions(start, end):
        out, attn = model(X_test[start:end])
        predictions = np.argmax(out, axis=-1)

        return predictions, attn

    batch_size = 128  # set it by your GPU size

    full_predictions = []
    for i in range(X_test.shape[0] // batch_size):
        y, _ = get_predictions(i * batch_size, (i + 1) * batch_size)
        full_predictions.append(y)

    y, attn = get_predictions((i + 1) * batch_size, X_test.shape[0])
    full_predictions.append(y)

    full_predictions = np.concatenate(full_predictions, axis=0)
    print(full_predictions.shape)

    full_predictions = full_predictions.reshape(-1, full_predictions.shape[-1])
    print(full_predictions.shape)


def y_sequence(y, seq_len):
    y_seg = []
    for yy in y:
        y_seg.append([yy for _ in range(seq_len)])
    return np.array(y_seg)


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


if __name__ == '__main__':
    data_path = "/home/jsenadesouza/DA-healthy2patient/results/outcomes/dataset/t900_INTELLIGENT_ADAPT_PAIN_15wd_15drop_painprev.npz"
    num_folders = 1
    X, y, y_target = load_data(data_path, clin_variable_target="pain_score_class")
    yy_t = np.array([0 if x == "mild" else 1 for x in y_target])

    folders = split(y)
    for folder_idx in range(num_folders):
        labels2idx = {k: idx for idx, k in enumerate(np.unique(y_target))}

        train_idx = folders[folder_idx][0]
        test_idx = folders[folder_idx][1]
        train_data, train_labels, test_data, test_labels = X[train_idx], yy_t[train_idx], X[test_idx], yy_t[test_idx]
        train_labels = y_sequence(train_labels, seq_len=train_data.shape[1])
        test_labels = y_sequence(test_labels, seq_len=train_data.shape[1])
        # sweep_id = init_logger(train_data, train_labels, val_data, val_labels)
        # wandb.agent(sweep_id, train_wandb, count=32)
        model = train(train_data, train_labels, test_data, test_labels)

