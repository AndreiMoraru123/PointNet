import torch.nn as nn
import torch


class TNet(nn.Module):
    # paper: https://arxiv.org/pdf/1612.00593.pdf
    def __init__(self, num_points=1024, num_features=3):
        super(TNet, self).__init__()
        self.num_features = num_features
        self.conv1 = nn.Conv1d(self.num_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_features * self.num_features)
        self.relu = nn.ReLU()
        self.num_features = num_features

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(num_points)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.num_features, requires_grad=True).float().view(
            1,self.num_features * self.num_features).repeat(batch_size, 1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.num_features, self.num_features)
        return x
