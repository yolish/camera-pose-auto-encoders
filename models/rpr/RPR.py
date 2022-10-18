import torch
import torch.nn as nn
import torch.nn.functional as F
from models.posenet.PoseNet import PoseNet
from models.pose_encoder import MultiSCenePoseEncoder

class RPR(PoseNet):
    """
    A class to represent a baseline image/pae-based relative pose regressor
    """
    def __init__(self, config, backbone_path, pae_path):
        """
        Constructor
        """
        super(RPR, self).__init__(config, backbone_path)

        self.virtual_rpr = config.get("virtual_rpr")
        self.common_dim = config.get("rpr_common_dim")
        self.latent_dim = config.get("rpr_latent_dim")
        self.pae_dim = config.get("rpr_pae_dim")
        if self.virtual_rpr:
            self.x_latent_fc_common_pae = nn.Linear(self.pae_dim, self.common_dim)
            self.q_latent_fc_common_pae = nn.Linear(self.pae_dim, self.common_dim)

        self.x_latent_fc_common_img = nn.Linear(self.backbone_dim, self.common_dim)
        self.q_latent_fc_common_img = nn.Linear(self.backbone_dim, self.common_dim)

        self.x_rel_latent_mlp = nn.Sequential(nn.Linear(self.common_dim * 2, self.latent_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.latent_dim, self.latent_dim // 2),
                                             nn.ReLU())

        self.q_rel_latent_mlp = nn.Sequential(nn.Linear(self.common_dim * 2, self.latent_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.latent_dim, self.latent_dim // 2),
                                             nn.ReLU())

        # Regressor layers
        self.x_reg = nn.Linear(self.latent_dim // 2, 3)
        self.q_reg = nn.Linear( self.latent_dim // 2, 4)
        pd = config.get("rpr_dropout")
        if pd is None:
            pd = 0.0
        self.dropout = nn.Dropout(p=pd)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

        # Initialize PAE if pae-based
        if self.virtual_rpr:
            pose_encoder = MultiSCenePoseEncoder(config.get("hidden_dim"))
            pose_encoder.load_state_dict(torch.load(pae_path))
            self.pae = pose_encoder

    def encode_pose(self, pose, scene):
        return self.pae(pose, scene)

    def encode_img(self, img):
        if self.backbone_type == "efficientnet":
            latent = self.backbone.extract_features(img)
        else:
            latent = self.backbone(img)
        latent = self.avg_pooling_2d(latent).flatten(start_dim=1)
        latent_x = torch.nn.functional.relu(self.x_latent_fc_common_img(latent))
        latent_q = torch.nn.functional.relu(self.q_latent_fc_common_img(latent))
        return latent_x, latent_q

    def regress_rel_pose(self, query, latent_neighbor_x, latent_neighbor_q):
        latent_query_x, latent_query_q = self.encode_img(query)
        if self.virtual_rpr:
            latent_neighbor_x =  torch.nn.functional.relu(self.x_latent_fc_common_pae(latent_neighbor_x))
            latent_neighbor_q =torch.nn.functional.relu(self.q_latent_fc_common_pae(latent_neighbor_q))

        latent_rel_x = self.dropout(self.x_rel_latent_mlp(torch.cat((latent_query_x, latent_neighbor_x), dim=1)))
        latent_rel_q = self.dropout(self.q_rel_latent_mlp(torch.cat((latent_query_q, latent_neighbor_q), dim=1)))

        rel_x = self.x_reg(latent_rel_x)
        rel_q = self.q_reg(latent_rel_q)

        return {'rel_pose': torch.cat((rel_x, rel_q), dim=1)}


