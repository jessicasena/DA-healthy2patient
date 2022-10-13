import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):

    def   __init__(self, n_classes):
        super(CNNLSTM, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, kernel_size=10),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(True),

            nn.Conv1d(32, 64, kernel_size=10),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.BatchNorm1d(64)
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x_sequence):
        b_z, ts, c, f = x_sequence.shape
        ii = 0
        y = self.features(x_sequence[:, ii])
        y = torch.mean(y, dim=2)
        output, (hn, cn) = self.lstm(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.features(x_sequence[:, ii])
            y = torch.mean(y, dim=2)
            out, (hn, cn) = self.lstm(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc1(out)
        out = self.fc2(out)
        return out



class MiniMetaSenseModel(nn.Module):
    """
    Meta Sense model extracted from the paper Metasense: few-shot adaptation to
    untrained conditions in deep mobile sensing [1].

    Parameters
    ----------
    - n_classes: int
        Number of target classes.

    Notes
    -----
    - [1] Gong, T., Kim, Y., Shin, J. and Lee, S.J., 2019, November. Metasense: few-shot
    adaptation to untrained conditions in deep mobile sensing. In Proceedings of the 17th
    Conference on Embedded Networked Sensor Systems (pp. 110-123).
    """

    def __init__(self, n_classes):
        super(MiniMetaSenseModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, kernel_size=10),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(True),

            nn.Conv1d(32, 64, kernel_size=10),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.BatchNorm1d(64)
        )

        self.classifier = nn.Sequential(

            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Linear(32, n_classes))

    def forward(self, x):
        x = self.features(x)
        y = torch.mean(x, dim=2)
        out = self.classifier(y)
        return out, y


class MetaSenseModel(nn.Module):
    """
    Meta Sense model extracted from the paper Metasense: few-shot adaptation to
    untrained conditions in deep mobile sensing [1].

    Parameters
    ----------
    - n_classes: int
        Number of target classes.

    Notes
    -----
    - [1] Gong, T., Kim, Y., Shin, J. and Lee, S.J., 2019, November. Metasense: few-shot
    adaptation to untrained conditions in deep mobile sensing. In Proceedings of the 17th
    Conference on Embedded Networked Sensor Systems (pp. 110-123).
    """

    def __init__(self, n_classes):
        super(MetaSenseModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=10),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 256, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 512, kernel_size=10),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Linear(64, n_classes))

    def forward(self, x1):
        x1 = self.features(x1)
        y = torch.mean(x1, dim=2)
        out = self.classifier(y)
        return out

class MetaSenseModeladdData(nn.Module):
    """
    Meta Sense model extracted from the paper Metasense: few-shot adaptation to
    untrained conditions in deep mobile sensing [1].

    Parameters
    ----------
    - n_classes: int
        Number of target classes.

    Notes
    -----
    - [1] Gong, T., Kim, Y., Shin, J. and Lee, S.J., 2019, November. Metasense: few-shot
    adaptation to untrained conditions in deep mobile sensing. In Proceedings of the 17th
    Conference on Embedded Networked Sensor Systems (pp. 110-123).
    """

    def __init__(self, n_classes, n_add_features):
        super(MetaSenseModeladdData, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=10),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 256, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 512, kernel_size=10),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
        )
        self.clinical_features = nn.Sequential(
            nn.Linear(n_add_features, n_add_features*2),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + (n_add_features*2), 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Linear(64, n_classes))

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.clinical_features(x2)
        y = torch.mean(x1, dim=2)
        combined = torch.cat((y, x2), dim=1)
        out = self.classifier(combined)
        return out


class ProtoMetaSenseModel(nn.Module):
    """
    Meta Sense model extracted from the paper Metasense: few-shot adaptation to
    untrained conditions in deep mobile sensing [1] adapted for protypicals networks.
    
    Notes
    -----
    - [1] Gong, T., Kim, Y., Shin, J. and Lee, S.J., 2019, November. Metasense: few-shot
    adaptation to untrained conditions in deep mobile sensing. In Proceedings of the 17th
    Conference on Embedded Networked Sensor Systems (pp. 110-123).
    """

    def __init__(self):
        super(ProtoMetaSenseModel, self).__init__()
        h = 8

        self.features = nn.Sequential(
            nn.Conv1d(3, h, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(h),

            nn.Conv1d(h, h, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(h),

            nn.Conv1d(h, h*2, kernel_size=3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(h*2),
            nn.ReLU(True),

            nn.Conv1d(h*2, h*2, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(h*2),

            nn.Conv1d(h*2, h*4, kernel_size=3),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(h*4),
        )

    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        return out

