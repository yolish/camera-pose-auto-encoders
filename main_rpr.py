"""
Entry point training and testing image-based and virtual RPR (for comparison)
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from datasets.RelPoseDataset import RelPoseDataset
from models.pose_losses import CameraPoseLoss
from os.path import join
from models.transposenet.EMSTransPoseNet import EMSTransPoseNet
from models.rpr.RPR import RPR
import transforms3d as t3d


def compute_rel_pose(poses1, poses2):
    # p1 p_rel = p2
    rel_pose = torch.zeros_like(poses1)
    poses1 = poses1.cpu().numpy()
    poses2 = poses2.cpu().numpy()
    for i, p1 in enumerate(poses1):
        p2 = poses2[i]
        t1 = p1[:3]
        q1 = p1[3:]
        rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

        t2 = p2[:3]
        q2 = p2[3:]
        rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

        t_rel = t2 - t1
        rot_rel = np.dot(np.linalg.inv(rot1), rot2)
        q_rel = t3d.quaternions.mat2quat(rot_rel)
        rel_pose[i][:3] = torch.Tensor(t_rel).to(device)
        rel_pose[i][3:] = torch.Tensor(q_rel).to(device)

    return rel_pose

def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)
    return q

def compute_abs_pose_torch(rel_pose, abs_pose_neighbor):
    abs_pose_query = torch.zeros_like(rel_pose)
    abs_pose_query[:, :3] = abs_pose_neighbor[:, :3] + rel_pose[:, :3]
    abs_pose_query[:, 3:] = qmult(abs_pose_neighbor[:, 3:], rel_pose[:, 3:])
    return abs_pose_query

def compute_abs_pose(rel_pose, abs_pose_neighbor, device):
    # p_neighbor p_rel = p_query
    # p1 p_rel = p2
    abs_pose_query = torch.zeros_like(rel_pose)
    rel_pose = rel_pose.cpu().numpy()
    abs_pose_neighbor = abs_pose_neighbor.cpu().numpy()
    for i, rpr in enumerate(rel_pose):
        p1 = abs_pose_neighbor[i]

        t_rel = rpr[:3]
        q_rel = rpr[3:]
        rot_rel = t3d.quaternions.quat2mat(q_rel/ np.linalg.norm(q_rel))

        t1 = p1[:3]
        q1 = p1[3:]
        rot1 = t3d.quaternions.quat2mat(q1/ np.linalg.norm(q1))

        t2 = t1 + t_rel
        rot2 = np.dot(rot1,rot_rel)
        q2 = t3d.quaternions.mat2quat(rot2)
        abs_pose_query[i][:3] = torch.Tensor(t2).to(device)
        abs_pose_query[i][3:] = torch.Tensor(q2).to(device)

    return abs_pose_query


def get_closest_sample(poses, scene_ids, dataset, start_index, sample_size=None, single_scene=False):
    samples = []
    poses = poses.cpu().numpy()
    if not single_scene:
        scene_ids = scene_ids.cpu().numpy()
    for i, p in enumerate(poses):
        # get the pose indices of the relevant scene
        if not single_scene:
            indices = np.where(np.array(dataset.scenes_ids) == scene_ids[i])[0]
        else:
            indices = list(range(len(dataset.poses)))
        # get the poses
        ref_poses = dataset.poses[indices]
        dist_x = np.linalg.norm(p[:3] - ref_poses[:, :3], axis=1)
        dist_x = dist_x / np.max(dist_x)
        dist_q = np.linalg.norm(p[3:] - ref_poses[:, 3:], axis=1)
        dist_q = dist_q / np.max(dist_q)
        sorted = np.argsort(dist_x + dist_q)

        if sample_size is not None:
            closest_index = np.random.choice(sorted[start_index:(start_index+sample_size)], size=1)[0]
        else:
            closest_index = sorted[start_index]
        neighbor_sample = dataset[indices[closest_index]]
        samples.append(neighbor_sample)
    return samples

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("rpr_backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file")
    arg_parser.add_argument("--apr_checkpoint_path",
                            help="path to a pre-trained APR model")
    arg_parser.add_argument("--apr_backbone_path", help="path to the APR backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("--pae_checkpoint_path",
                            help="path to a pre-trained PAE model (should match the APR model")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained RPR model")
    arg_parser.add_argument("--ref_poses_file", help="path to file with train poses")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} experiment for RPR".format(args.mode))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.fdeterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Image vs virtual-based RPR
    rpr_type = "image-based"
    do_pae_rpr = config.get("pae_rpr")
    if do_pae_rpr: # Load the PAE
        rpr_type = "virtual"

    # Create the RPR model - pae- or image-based encoder
    model = RPR(config, args.rpr_backbone_path, args.pae_checkpoint_path).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        transform = utils.train_transforms.get('baseline')
        train_dataset = RelPoseDataset(args.dataset_path, args.labels_file, transform)

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(train_dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        # Resetting temporal loss used for logging
        running_loss = 0.0
        n_samples = 0

        for epoch in range(n_epochs):
            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_rel_poses = minibatch['rel_pose'].to(dtype=torch.float32)
                batch_size = gt_rel_poses.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                neighbor_poses = minibatch['pose2'].to(device).to(dtype=torch.float32)

                # Estimate the relative pose
                # Zero the gradients
                optim.zero_grad()

                if do_pae_rpr:
                    # Encode poses with known scene
                    gt_scenes = minibatch['scene_id2'].to(device).to(dtype=torch.int64)
                    neighbor_latent_x, neighbor_latent_q = model.encode_pose(neighbor_poses,
                                                                        gt_scenes.unsqueeze(1).to(dtype=torch.float32))
                else: # image based
                    neighbor_imgs = minibatch['img2'].to(device).to(torch.float32)
                    neighbor_latent_x, neighbor_latent_q = model.encode_img(neighbor_imgs)

                est_rel_poses = model.regress_rel_pose(minibatch['img1'], neighbor_latent_x, neighbor_latent_q).get('rel_pose')

                criterion = pose_loss(est_rel_poses, gt_rel_poses)
                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_rel_poses.detach(), gt_rel_poses.detach())
                    msg = "[Batch-{}/Epoch-{}] running relative camera pose loss: {:.3f}, camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_freq_print),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item())
                    posit_err, orient_err = utils.pose_err(neighbor_poses.detach(), minibatch['pose1'].to(dtype=torch.float32).detach())
                    msg = msg + ", distance from neighbor images: {:.2f}[m], {:.2f}[deg]".format(posit_err.mean().item(),
                                                                        orient_err.mean().item())
                    logging.info(msg)
                    # Resetting temporal loss used for logging
                    running_loss = 0.0
                    n_samples = 0

            # Save checkpoint3n
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_{}_rpr_checkpoint-{}.pth'.format(rpr_type, epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_{}_rpr_final.pth'.format(rpr_type))


    else: # Test
        # APR model
        apr = EMSTransPoseNet(config, args.apr_backbone_path)
        apr.load_state_dict(torch.load(args.apr_checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.apr_checkpoint_path))
        apr.to(device)
        apr.eval()

        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        test_dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, False)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)

        train_dataset = CameraPoseDataset(args.dataset_path, args.ref_poses_file, transform, False, not do_pae_rpr)

        time_stats_pae = np.zeros(len(dataloader.dataset))
        time_stats_retrieval = np.zeros(len(dataloader.dataset))
        time_stats_rpr = np.zeros(len(dataloader.dataset))
        abs_stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                minibatch['scene'] = None # avoid using ground-truth scene during prediction

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # Forward pass to predict the pose
                t0 = time.time()
                res = apr(minibatch)
                init_est_pose = res.get('pose')
                scene_dist = res.get('scene_log_distr')
                scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32).unsqueeze(1)

                tic = time.time()
                # get closest pose / image
                neighbor_sample = get_closest_sample(init_est_pose, scene, train_dataset, start_index=0, single_scene=True)[0]
                closest_pose = torch.Tensor(neighbor_sample['pose']).unsqueeze(0).to(device).to(torch.float32)
                time_stats_retrieval[i] = (time.time() - tic) * 1000

                # Encode the pose or the image
                if do_pae_rpr:
                    tic = time.time()
                    latent_x, latent_q = model.encode_pose(closest_pose, scene)
                    torch.cuda.synchronize()
                    time_stats_pae[i] = (time.time() - tic)*1000
                    tic = time.time()
                else:
                    tic = time.time()
                    closest_img = neighbor_sample['img'].unsqueeze(0).to(device)
                    latent_x, latent_q = model.encode_img(closest_img)

                # Regress the relative pose
                res = model.regress_rel_pose(minibatch['img'], latent_x, latent_q)
                est_rel_pose = res['rel_pose']

                # Flip to get the relative from neighbor to query
                est_rel_pose[:, :3] = -est_rel_pose[:, :3]
                est_rel_pose[:, 4:] = -est_rel_pose[:, 4:]

                est_pose = compute_abs_pose(est_rel_pose, closest_pose, device)

                torch.cuda.synchronize()
                tn = time.time()
                time_stats_rpr[i] = (tn - tic)*1000

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                abs_stats[i, 0] = posit_err.item()
                abs_stats[i, 1] = orient_err.item()
                abs_stats[i, 2] = (tn - t0)*1000


                logging.info("Absolute Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    abs_stats[i, 0],  abs_stats[i, 1],  abs_stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median absolute pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(abs_stats[:, 0]), np.nanmedian(abs_stats[:, 1])))
        logging.info("Mean pose inference time:{:.2f}[ms]".format(np.mean(abs_stats[:, 2])))
        logging.info("Mean retrieval inference time:{:.2f}[ms]".format(np.mean(time_stats_retrieval)))
        logging.info("Mean PAE inference time:{:.2f}[ms]".format(np.mean(time_stats_pae)))
        logging.info("Mean {} RPR inference time:{:.2f}[ms]".format(rpr_type, np.mean(time_stats_rpr)))






