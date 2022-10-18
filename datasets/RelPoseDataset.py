import random
from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np


class RelPoseDataset(Dataset):
    def __init__(self, data_path, pairs_file, transform=None):
        self.img_path1, self.scenes1, self.scene_ids1, self.poses1, \
        self.img_path2, self.scenes2, self.scene_ids2, self.poses2, self.rel_poses = \
            read_pairs_file(data_path, pairs_file)
        self.transform = transform

    def __len__(self):
        return len(self.img_path1)

    def __getitem__(self, idx):
        img1 = imread(self.img_path1[idx])
        img2 = imread(self.img_path2[idx])
        pose1 = self.poses1[idx]
        pose2 = self.poses2[idx]
        rel_pose = self.rel_poses[idx]
        scene1 = self.scene_ids1[idx]
        scene2 = self.scene_ids2[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # randomly flip images in an image pair
        if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            pose1, pose2 = pose2, pose1
            scene1, scene2 = scene2, scene1
            rel_pose[:3] = -rel_pose[:3]
            rel_pose[3:] = [rel_pose[3], -rel_pose[4], -rel_pose[5], -rel_pose[6]]

        return {'img1': img1,
                'img2': img2,
                'pose1': pose1,
                'pose2': pose2,
                "scene_id1": scene1,
                "scene_id2": scene2,
                'rel_pose':rel_pose}

def read_pairs_file(dataset_path, labels_file):
    df = pd.read_csv(labels_file)
    img_paths = []
    scenes = []
    scene_ids = []
    all_poses = []
    n = df.shape[0]
    for suffix in ["a", "b"]:
        img_paths.append([join(dataset_path, path) for path in df['img_path_{}'.format(suffix)].values])
        scenes.append(df['scene_{}'.format(suffix)].values)
        scene_ids.append( df['scene_id_{}'.format(suffix)].values)
        poses = np.zeros((n, 7))
        poses[:, 0] = df['x1_{}'.format(suffix)].values
        poses[:, 1] = df['x2_{}'.format(suffix)].values
        poses[:, 2] = df['x3_{}'.format(suffix)].values
        poses[:, 3] = df['q1_{}'.format(suffix)].values
        poses[:, 4] = df['q2_{}'.format(suffix)].values
        poses[:, 5] = df['q3_{}'.format(suffix)].values
        poses[:, 6] = df['q4_{}'.format(suffix)].values
        all_poses.append(poses)
    img_paths1, img_paths2 = img_paths
    scenes1, scenes2 = scenes
    scene_ids1, scene_ids2 = scene_ids
    poses1, poses2 = all_poses
    rel_poses = np.zeros((n, 7))
    suffix = "ab"
    rel_poses[:, 0] = df['x1_{}'.format(suffix)].values
    rel_poses[:, 1] = df['x2_{}'.format(suffix)].values
    rel_poses[:, 2] = df['x3_{}'.format(suffix)].values
    rel_poses[:, 3] = df['q1_{}'.format(suffix)].values
    rel_poses[:, 4] = df['q2_{}'.format(suffix)].values
    rel_poses[:, 5] = df['q3_{}'.format(suffix)].values
    rel_poses[:, 6] = df['q4_{}'.format(suffix)].values

    return img_paths1, scenes1, scene_ids1, poses1, img_paths2, scenes2, scene_ids2, poses2, rel_poses