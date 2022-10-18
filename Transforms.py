# Callable transforms for use with the PyTorch DataLoader

import numpy as np
import torch


class PointSampler(object):
    """Sample a fixed number of points from a point cloud"""

    def __init__(self, num_points):
        self.num_points = num_points

    @staticmethod
    def area(pt1, pt2, pt3):
        # https://www.mathopenref.com/coordtrianglearea.html
        return 0.5 * np.linalg.norm(np.cross(pt2 - pt1, pt3 - pt1))

    @staticmethod
    def barycentric(pt1, pt2, pt3):
        # https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        u = np.random.rand()
        v = np.random.rand()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        w = 1 - u - v
        return u * pt1 + v * pt2 + w * pt3

    def __call__(self, sample):
        verts, faces = sample
        verts = np.array(verts)
        # sample points from faces
        face_areas = np.array([self.area(verts[face[0]], verts[face[1]], verts[face[2]]) for face in faces])
        face_areas /= face_areas.sum()

        face_indices = np.random.choice(len(faces), size=self.num_points, p=face_areas)
        face_points = np.random.rand(self.num_points, 3)
        face_points = np.array(
            [self.barycentric(verts[faces[face_index][0]], verts[faces[face_index][1]], verts[faces[face_index][2]]) for
             face_index, face_point in zip(face_indices, face_points)])

        return face_points


class PointCloudToTensor(object):
    """Convert a point cloud to a tensor"""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()


class Normalize(object):
    """Normalize a point cloud"""

    def __call__(self, sample):
        return (sample - sample.mean(axis=0)) / sample.std(axis=0)


class RandomRotation(object):
    """Rotate a point cloud"""

    def __call__(self, sample, axis='Z'):

        theta = np.random.rand() * 2 * np.pi
        # axis = np.random.choice(['X', 'Y', 'Z'])

        if axis == 'Z':
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
        elif axis == 'X':
            R = np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])
        elif axis == 'Y':
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

        return np.dot(sample, R)


class RandomScale(object):
    """Scale a point cloud"""

    def __call__(self, sample):
        scale = np.random.rand() + 0.5
        return sample * scale


class RandomTranslation(object):
    """Translate a point cloud"""

    def __call__(self, sample):
        translation = np.random.rand(3) - 0.5
        return sample + translation


class RandomNoise(object):
    """Add noise to a point cloud"""

    def __call__(self, sample):
        noise = np.random.rand(*sample.shape) - 0.5
        return sample + noise