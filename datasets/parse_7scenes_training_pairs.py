from datasets.CameraPoseDataset import read_labels_file
from os.path import join
from collections import defaultdict
import numpy as np
import transforms3d as t3d
import re
import cv2
import random

def compute_rel_pose(p1, p2):
    t1 = p1[:3]
    q1 = p1[3:]
    rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

    t2 = p2[:3]
    q2 = p2[3:]
    rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

    t_rel = t2 - t1
    rot_rel = np.dot(np.linalg.inv(rot1), rot2)
    q_rel = t3d.quaternions.mat2quat(rot_rel)
    return t_rel, q_rel


def parse_pose_file(file_path):
        f = open(file_path, 'r')
        lines = f.readlines()[0:4]
        f.close()

        aff_mat = np.zeros((4, 4))
        pattern = re.compile(r"[e\.\-\+\d]+")

        for i, line in enumerate(lines):
            matches = pattern.findall(line)
            for j, match in enumerate(matches):
                aff_mat[i,j] = float(match)

        t, rotm_R, _, _ = t3d.affines.decompose(aff_mat)
        q = t3d.quaternions.mat2quat(rotm_R)

        pose = np.zeros(7)
        pose[:3] = t
        pose[3:] = q
        return pose

def reprojection(depth, r1, r2, t1, t2, h=480, w=640):
    # function from CamNet basecode
    K = np.mat([[585,0,320],[0,585,240],[0,0,1]])
    x_list = []
    depth_list = []
    for i in range(0,h,10):
        for j in range(0,w,10):
            if depth[i][j]!=0 and depth[i][j]!=65535:
                x_list.append([j,i,1])
                depth_list.append(depth[i][j])
    x1 = np.mat(x_list)
    dep = np.mat(depth_list)
    x_new = np.multiply((K.I.dot(x1.T)),(dep/1000))
    x_new1 = np.mat(r2).I.dot(np.mat(r1).dot(x_new)+ (np.mat(t1)-np.mat(t2)).T)
    result = np.array(K.dot(x_new1))
    x2 = result/result[2]
    flag = 0
    for i in range(len(x_list)):
        if (x2[0][i] > 0) and (x2[0][i] < w) and (x2[1][i] > 0) and (x2[1][i] < h):
            flag += 1
    #print(x_list[0],x2[:,0])
    return float(flag)/len(x_list)

def relative_r(a1, a2):
    a1 = np.mat(a1)
    a2 = np.mat(a2)
    return np.arccos((np.trace(a1.I.dot(a2))-1)/2)*180/np.pi

labels_file = "7Scenes/7scenes_all_scenes.csv"
dataset_path = "/mnt/data/7Scenes/7Scenes"
pairs_file = "nnnet_trainining_pairs.txt" # download from NNNet repo
# get all the images their poses and scenes
img_paths, poses, scenes, scenes_ids = read_labels_file(labels_file, dataset_path)

out_file = open("7scenes_training_pairs.csv", 'w')
out_file.write("img_path_a,scene_a,scene_id_a,x1_a,x2_a,x3_a,q1_a,q2_a,q3_a,q4_a,"
               "img_path_b,scene_b,scene_id_b,x1_b,x2_b,x3_b,q1_b,q2_b,q3_b,q4_b,"
               "x1_ab,x2_ab,x3_ab,q1_ab,q2_ab,q3_ab,q4_ab\n")

# read the pairs file and get the poses
# write new file
#missing = {}
scenes_dict = defaultdict(str)

for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
    scenes_dict[i] = scene
fnames1, fnames2, t_gt, q_gt = [], [], [], []
with open(pairs_file, 'r') as f:
    for line_idx, line in enumerate(f):
        chunks = line.rstrip().split(' ')
        scene_id = int(chunks[2])
        img1_path = join(dataset_path, scenes_dict[scene_id], chunks[0][1:])
        try:
            idx1 = img_paths.index(img1_path)
            scene1 = scenes[idx1]
            scene_id1 = scenes_ids[idx1]
            pose1 = poses[idx1]
        except:
            # ignore pairs from head seq01 since they belong to the test set
            continue

        out_file.write("{},{},{},{},".format(img1_path.replace(dataset_path,""),
                                      scene1, scene_id1, ",".join(["{}".format(pose1[i]) for i in range(7)])))

        img2_path = join(dataset_path, scenes_dict[scene_id], chunks[1][1:])
        try:
            idx2 = img_paths.index(img2_path)
            scene2 = scenes[idx2]
            scene_id2 = scenes_ids[idx2]
            pose2 = poses[idx2]
        except:
            # ignore NNnet pairs from head seq01 since they belong to the test set
            continue


        out_file.write("{},{},{},{},".format(img2_path.replace(dataset_path, ""),
                                      scene2, scene_id2, ",".join(["{}".format(pose2[i]) for i in range(7)])))

        # for sanity
        t_rel, q_rel = compute_rel_pose(pose1, pose2)

        t_rel_from_file = [float(chunks[3]), float(chunks[4]), float(chunks[5])]
        q_rel_from_file = [float(chunks[6]),
                                       float(chunks[7]),
                                       float(chunks[8]),
                                       float(chunks[9])]

        out_file.write(",".join(["{}".format(t_rel_from_file[i]) for i in range(3)]))
        out_file.write(",")
        out_file.write(",".join(["{}".format(q_rel_from_file[i]) for i in range(4)]))
        out_file.write("\n")

for s in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
    print(s)
    scene_indices = np.where(scenes == s)[0]
    n = len(scene_indices)
    for j in range(n):
        idx1 = scene_indices[j]
        pose1 = poses[idx1]
        img1_path = img_paths[idx1]
        scene1 = s
        scene_id1 = scenes_ids[idx1]
        depth1 = cv2.imread(img1_path.replace('pose.txt','depth.png'),2)
        for k in range(random.randint(0,n//100), n,n//100):
            # compute the overlap rate (depth-wise)
            idx2 = scene_indices[k]
            pose2 = poses[idx2]
            img2_path = img_paths[idx2]
            scene2 = s
            scene_id2 = scenes_ids[idx2]
            depth2 = cv2.imread(img2_path.replace('pose.txt', 'depth.png'), 2)
            rate1 = reprojection(depth1, t3d.quaternions.quat2mat(pose1[3:]),
                                 t3d.quaternions.quat2mat(pose2[3:]), pose1[:3], pose2[:3])
            rate2 = reprojection(depth2, t3d.quaternions.quat2mat(pose2[3:]),
                                 t3d.quaternions.quat2mat(pose1[3:]), pose2[:3], pose1[:3])
            rater = relative_r(t3d.quaternions.quat2mat(pose1[3:]), t3d.quaternions.quat2mat(pose2[3:]))
            # add if meet condition
            if (rate1> 0.4 and rate2>0.4 and rater< 30) or (rate1> 0.3 and rate2>0.3 and rater> 60) \
                or (rate1< 0.25 and rate2<0.25 and rate1 > 0.05 and rate2 > 0.05):
                print("{}-{}".format(img1_path,img2_path))
                out_file.write("{},{},{},{},".format(img1_path.replace(dataset_path, ""),
                                                     scene1, scene_id1,
                                                     ",".join(["{}".format(pose1[i]) for i in range(7)])))
                out_file.write("{},{},{},{},".format(img2_path.replace(dataset_path, ""),
                                                     scene2, scene_id2,
                                                     ",".join(["{}".format(pose2[i]) for i in range(7)])))

                # for sanity
                t_rel, q_rel = compute_rel_pose(pose1, pose2)

                out_file.write(",".join(["{}".format(t_rel[i]) for i in range(3)]))
                out_file.write(",")
                out_file.write(",".join(["{}".format(q_rel[i]) for i in range(4)]))
                out_file.write("\n")

out_file.close()










