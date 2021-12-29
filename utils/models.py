import torch
import torch.nn as nn


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
            nn.Conv1d(3, 16, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, kernel_size=3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(True),

            nn.Conv1d(32, 64, kernel_size=3),
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
            nn.Conv1d(3, 32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 512, kernel_size=3),
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

    def forward(self, x):
        x = self.features(x)
        y = torch.mean(x, dim=2)
        out = self.classifier(y)
        return out, y


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

