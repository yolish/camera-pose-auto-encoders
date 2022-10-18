import argparse
import torch
import numpy as np
import json
import logging
from util import utils
from datasets.CameraPoseDataset import CameraPoseDataset
from os.path import join
from models.pose_encoder import PoseEncoder, MultiSCenePoseEncoder
import torch.nn as nn
import matplotlib.pyplot as plt


class Decoder(torch.nn.Module):
    def __init__(self, dim, img_size):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(Decoder, self).__init__()
        dim = dim*2
        self.img_size = img_size
        self.decoder = torch.nn.Sequential(nn.Linear(dim,512),
                                             nn.ReLU(),
                                             nn.Linear(512, 1024),
                                             nn.ReLU(),
                                             nn.Linear(1024, 2048),
                                             nn.ReLU(),
                                             nn.Linear(2048, self.img_size*self.img_size*3),
                                             nn.Sigmoid()
                                             )

        self._reset_params()

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, latent):
        batch_size = latent.shape[0]
        out = self.decoder(latent)
        return out.reshape((batch_size, 3, self.img_size, self.img_size))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or demo")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file")
    arg_parser.add_argument("encoder_checkpoint_path", help="path to a trained pose encoder")
    arg_parser.add_argument("--decoder_checkpoint_path", help="path to apr refiner component")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} image reconstruction from encoding".format(args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config['decoder']
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)


    is_single_scene = config.get("single_scene")
    if is_single_scene:
        pose_encoder = PoseEncoder(config.get("hidden_dim")).to(device)
    else:
        pose_encoder = MultiSCenePoseEncoder(config.get("hidden_dim")).to(device)
    pose_encoder.load_state_dict(torch.load(args.encoder_checkpoint_path, map_location=device_id))
    logging.info("Initializing encoder from checkpoint: {}".format(args.encoder_checkpoint_path))
    pose_encoder.eval()

    img_size = config.get("img_size")
    decoder = Decoder(config.get("hidden_dim"), img_size).to(device)
    if args.decoder_checkpoint_path is not None:
        decoder.load_state_dict(torch.load(args.decoder_checkpoint_path, map_location=device_id))
        logging.info("Initializing encoder from checkpoint: {}".format(args.decoder_checkpoint_path))

    transform = utils.get_base_transform(img_size)

    if args.mode == 'train':
        # Set to train mode
        decoder.train()

        # Set the losses
        l1_loss = torch.nn.L1Loss().to(device)

        # Set the optimizer and scheduler
        optim = torch.optim.Adam(decoder.parameters(),
                                  lr=config.get('lr'), eps=config.get("eps"), weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, False)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        debug_display = False
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.09
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene').to(device).to(dtype=torch.float32).unsqueeze(1)
                img = minibatch.get('img').to(device)
                minibatch['scene'] = None
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                with torch.no_grad():
                    if not is_single_scene:
                        latent_x, latent_q = pose_encoder(gt_pose, gt_scene)
                    else:
                        latent_x, latent_q = pose_encoder(gt_pose)

                rec_img = decoder(torch.cat((latent_x, latent_q), dim=1))

                criterion = l1_loss(rec_img, img)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    logging.info("[Batch-{}/Epoch-{}] running MSE loss: {:.3f}".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples)))
                    if debug_display:
                        rec_np_img = rec_img.detach().cpu()[0].squeeze(0).permute((1,2,0)).numpy() * 255.0
                        plt.imshow(rec_np_img.astype(np.int32))
                        plt.show()

                        np_img = img.detach().cpu()[0].squeeze(0).permute((1,2,0)).numpy() * 255.0
                        plt.imshow(np_img.astype(np.int32))
                        plt.show()
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(decoder.state_dict(), checkpoint_prefix + '_decoder_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(decoder.state_dict(), checkpoint_prefix + '_decoder_final.pth'.format(epoch))


    else: # Demo

        # Set to eval mode
        decoder.eval()

        # Set the dataset and data loader
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        #indices = [10,200] # heads
        indices = [1, 6]  # shop facade
        with torch.no_grad():
            for i in indices:
                sample = dataset[i]
                scene = torch.Tensor([sample.get('scene')]).to(device).to(dtype=torch.float32).unsqueeze(1)
                pose = torch.Tensor(sample.get('pose')).to(dtype=torch.float32).to(device).unsqueeze(0)
                img = sample.get('img')

                if not is_single_scene:
                    latent_x, latent_q = pose_encoder(pose, scene)
                else:
                    latent_x, latent_q = pose_encoder(pose)

                rec_img = decoder(torch.cat((latent_x, latent_q), dim=1))
                # plot here img vs rec_img

                rec_np_img = rec_img.cpu()[0].squeeze(0).permute((1, 2, 0)).numpy() * 255.0
                plt.imshow(rec_np_img.astype(np.int32))
                #plt.show()
                plt.savefig("{}_recon.png".format(i))

                np_img = img.permute((1, 2, 0)).numpy() * 255.0
                plt.imshow(np_img.astype(np.int32))
                #plt.show()
                plt.savefig("{}_orig.png".format(i))




