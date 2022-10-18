from SpatialTransformer import *
from TNet import *


class SegmentationNetwork(nn.Module):
    def __init(self, k=2):
        super(SegmentationNetwork, self).__init()
        self.k = k
        self.stn = SpatialTransformer()

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def forward(self, x):
        x, _, _ = self.stn(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x
