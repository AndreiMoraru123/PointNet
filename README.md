# PointNet

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green) ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white) ![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)

Implementation of PointNet for 3D Point Cloud classification

![pointnet](https://user-images.githubusercontent.com/81184255/197140870-01b273c2-3ad8-456f-b4c3-75f235ccb61e.png)


```bibtex
@misc{https://doi.org/10.48550/arxiv.1612.00593,
  doi = {10.48550/ARXIV.1612.00593},  
  url = {https://arxiv.org/abs/1612.00593},
  author = {Qi, Charles R. and Su, Hao and Mo, Kaichun and Guibas, Leonidas J.},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  publisher = {arXiv},
  year = {2016},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

Original Repo is here: https://github.com/fxia22/pointnet.pytorch

and here is the [medium article](https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263) that I followed for the implementation.

### Train

```bash
python train.py
```

### Evaluate

```bash
python inference.py
```


### Trained and tested on [ModelNet10](https://modelnet.cs.princeton.edu) for the 10 classes

```
bathtub: 0
bed: 1
chair: 2
desk: 3
dresser: 4
monitor: 5
night_stand: 6
sofa: 7
table: 8
toilet: 9
```

![face](https://user-images.githubusercontent.com/81184255/196343885-c14d4394-e08b-4b72-b602-b75199db0663.gif)

![pointcloud_min](https://user-images.githubusercontent.com/81184255/196346649-2502cf31-e3d2-41e7-a403-da0919e571d7.gif)



# Model

![image](https://user-images.githubusercontent.com/81184255/196339977-896052a6-1b98-4b36-84bf-cfdf99992eb3.png)

```python

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

```


## T-Net

![image](https://user-images.githubusercontent.com/81184255/196340036-de6f8f62-c6c4-4629-9207-0e90994af806.png)

```python

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
```

![image](https://user-images.githubusercontent.com/81184255/196340854-ff1887c2-0cce-4d5e-8964-4e2d68ce89eb.png)

```bibtex
@unknown{unknown,
author = {Gutierrez Becker, Benjamín and Wachinger, Christian},
year = {2018},
month = {06},
pages = {},
title = {Deep Multi-Structural Shape Analysis: Application to Neuroanatomy}
}
```

## Spatial Transformer 

![image](https://user-images.githubusercontent.com/81184255/196341003-ea737e76-ab43-47d3-b63e-95e502f79d7e.png)

```python
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
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.1506.02025,
  doi = {10.48550/ARXIV.1506.02025},
  url = {https://arxiv.org/abs/1506.02025},
  author = {Jaderberg, Max and Simonyan, Karen and Zisserman, Andrew and Kavukcuoglu, Koray},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Spatial Transformer Networks},
  publisher = {arXiv},
  year = {2015},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

# Results:

## Correct predictions:


![Figure_1](https://user-images.githubusercontent.com/81184255/196341383-9b728de5-738b-4a3a-8dd2-877d1876ee68.png)

![Figure_2](https://user-images.githubusercontent.com/81184255/196341416-06b0c49a-9004-456d-a3c1-e232fb0dff69.png)

![Figure_3](https://user-images.githubusercontent.com/81184255/196341447-582a7799-9380-4f11-a64e-3abf263af0e1.png)

![Figure_4](https://user-images.githubusercontent.com/81184255/196341485-24601955-d80e-49cc-9b87-2167370d89f8.png)

![Figure_6](https://user-images.githubusercontent.com/81184255/196341518-61b3a726-7690-476d-b931-e6ff92ab97c1.png)

![Figure_11](https://user-images.githubusercontent.com/81184255/196341631-096c58b0-c759-4e8d-adda-c23d2a63ac92.png)

![Figure_14](https://user-images.githubusercontent.com/81184255/196341660-3db4507b-55a2-4670-b07b-a5a7e2520483.png)

![Figure_12](https://user-images.githubusercontent.com/81184255/196342149-70cc3e3b-ad3f-4cab-83c5-3134cc5f7971.png)

![Figure_15](https://user-images.githubusercontent.com/81184255/196342182-b22cf965-38ab-4621-8f38-66d66cd46f6f.png)



## Misnomers:


![Figure_16](https://user-images.githubusercontent.com/81184255/196341599-fa253cd8-bce7-4902-99c1-ceb90b954f73.png)

![Figure_10](https://user-images.githubusercontent.com/81184255/196341710-d65960db-65f8-4208-9113-7dd0ca5c6fe3.png)

![Figure_5](https://user-images.githubusercontent.com/81184255/196341731-693454d5-8b64-4955-8a3d-68798fa2fdf3.png)

![Figure_8](https://user-images.githubusercontent.com/81184255/196341767-74051dba-3561-48f1-b5d0-cffe91205e58.png)

![Figure_13](https://user-images.githubusercontent.com/81184255/196341779-c00245f0-2d0e-4fe2-b218-a0f86d44e8aa.png)

![Figure_9](https://user-images.githubusercontent.com/81184255/196341812-cc403e42-ec1e-42e9-8dc7-9245be3a764f.png)
