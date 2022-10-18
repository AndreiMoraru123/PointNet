from SpatialTransformer import *


class PointNet(nn.Module):

    def __init__(self, num_points=1024, num_classes=10, scores=2):
        super(PointNet, self).__init__()
        self.stn = SpatialTransformer()
        self.scores = scores
        self.num_points = num_points
        self.num_classes = num_classes

        # classification head
        self.fc1 = nn.Linear(self.num_points, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

        # segmentation head
        # self.conv1 = nn.Conv1d(1088, 512, 1)
        # self.conv2 = nn.Conv1d(512, 256, 1)
        # self.conv3 = nn.Conv1d(256, 128, 1)
        # self.conv4 = nn.Conv1d(128, self.scores, 1)
        #
        # self.bn3 = nn.BatchNorm1d(512)
        # self.bn4 = nn.BatchNorm1d(256)
        # self.bn5 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_cls, tr3x3, tr64x64 = self.stn(x)
        x_cls = self.relu(self.bn1(self.fc1(x_cls)))
        x_cls = self.relu(self.bn2(self.fc2(x_cls)))
        x_cls = self.dropout(x_cls)
        x_cls = self.fc3(x_cls)

        # segmentation head
        # x_seg = x.view(-1, self.num_points, 1).repeat(1, 1, self.num_points)
        # x_seg = torch.cat([x_seg, x], dim=1)
        # x_seg = self.relu(self.bn3(self.conv1(x_seg)))
        # x_seg = self.relu(self.bn4(self.conv2(x_seg)))
        # x_seg = self.relu(self.bn5(self.conv3(x_seg)))
        # x_seg = self.conv4(x_seg)

        return x_cls, tr3x3, tr64x64



