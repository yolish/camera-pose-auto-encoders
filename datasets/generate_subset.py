from datasets.CameraPoseDataset import read_labels_file
import numpy as np
import pandas as pd

p = 0.3
labels_file = "7Scenes/7scenes_all_scenes.csv"
dataset_path = "/mnt/data/7Scenes/7Scenes"

img_paths, poses, scenes, scenes_ids = read_labels_file(labels_file, dataset_path)
scene_to_poses = {}
indices = []
np.unique(scenes_ids)
for scene_id in np.unique(scenes_ids):
    scene_indices = np.where(scenes_ids == scene_id)[0]
    selected_indices = np.random.choice(scene_indices, size = int(p*len(scene_indices)))
    indices.append(selected_indices)
indices = np.concatenate(indices)
df = pd.read_csv(labels_file)
df = df.iloc[indices, :]
df.to_csv(labels_file+"_{}_subset.csv".format(p), index=False)

