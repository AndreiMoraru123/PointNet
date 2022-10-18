import torch

from TNet import *


class SpatialTransformer(nn.Module):
    """ Spatial Transformer Network """

    def __init__(self, num_points=1024):
        super(SpatialTransformer, self).__init__()
        self.num_points = num_points
        self.input_transform = TNet(num_points=self.num_points, num_features=3)
        self.feature_transform = TNet(num_points=self.num_points, num_features=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.mp1 = nn.MaxPool1d(num_points)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 3x3 transform
        tr3x3 = self.input_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), tr3x3).transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))

        # 64x64 transform
        tr64x64 = self.feature_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), tr64x64).transpose(1, 2)
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        return x, tr3x3, tr64x64
