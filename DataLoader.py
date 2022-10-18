import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from path import Path


class PointCloudDataset(Dataset):

    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.folders = [dir for dir in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, dir))]
        self.files = list()
        # read .off files in self.classes from ModelNet10 for each class
        self.classes = {folder: i for i, folder in enumerate(self.folders)}
        self.files = []
        for category in self.classes.keys():
            new_dir = root/Path(category)/mode
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = dict()
                    sample['category'] = category
                    sample['file'] = new_dir/file
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def read_off(file):
        if 'OFF' != file.readline().strip():
            raise Exception('Not a valid OFF header')
        n_verts, n_faces, _ = [int(s) for s in file.readline().split()]
        verts = list()
        for i in range(n_verts):
            verts.append([float(s) for s in file.readline().split()])
        faces = list()
        for i in range(n_faces):
            face = [int(s) for s in file.readline().split()]
            assert len(face) == 4
            faces.append(face[1:])

        return np.array(verts), np.array(faces)

    def process(self, file):
        verts, faces = self.read_off(file)
        if self.transform:
            pointcloud = self.transform((verts, faces))
        else:
            pointcloud = verts
        return pointcloud

    def __getitem__(self, idx):
        path = self.files[idx]['file']
        category = self.files[idx]['category']
        with open(path, 'r') as f:
            pointcloud = self.process(f)
        return {'pointcloud': pointcloud, 'category': self.classes[category]}

