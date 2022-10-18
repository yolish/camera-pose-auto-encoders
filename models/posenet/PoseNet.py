import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class PoseNet(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet) with a CNN backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self, config, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to the cnn backbone
        """
        super(PoseNet, self).__init__()

        backbone_type = config.get("rpr_backbone_type")
        if backbone_type == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path))
            self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone_dim = 2048

        elif backbone_type == "resnet34":
            backbone = torchvision.models.resnet34(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))
            self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone_dim = 512

        elif backbone_type == "mobilenet":
            backbone = torchvision.models.mobilenet_v2(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 1280

        elif backbone_type == "efficientnet":
            # Efficient net
            self.backbone = torch.load(backbone_path)
            backbone_dim = 1280
            self.standard_forward = True
        else:
            raise NotImplementedError("backbone type: {} not supported".format(backbone_type))

        self.latent_dim = config.get("hidden_dim") # 256
        self.backbone_type = backbone_type

        # Regressor layers
        self.x_latent_fc = nn.Linear(backbone_dim, self.latent_dim)
        self.q_latent_fc = nn.Linear(backbone_dim, self.latent_dim)
        self.x_reg = nn.Linear(self.latent_dim, 3)
        self.q_reg = nn.Linear( self.latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
        self.backbone_dim = backbone_dim

    def forward_body(self, data):
        u = data.get('img')
        if self.backbone_type == "efficientnet":
            u = self.backbone.extract_features(u)
        else:
            u = self.backbone(u)
        u = self.avg_pooling_2d(u)
        u = u.flatten(start_dim=1)
        latent_q = F.relu(self.x_latent_fc(u))
        latent_x = F.relu(self.q_latent_fc(u))
        return latent_x, latent_q

    def forward_heads(self, data):
        latent_x = data.get("latent_x")
        latent_q = data.get("latent_q")
        x = self.x_reg(self.dropout(latent_x))
        q = self.q_reg(self.dropout(latent_q))
        return {"pose": torch.cat((x, q), dim=1)}


    def forward(self, data):
        """
        Forward pass
        :param data: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7) and latents
        """
        latent_x, latent_q = self.forward_body(data)
        pose = self.forward_heads({"latent_x":latent_x, "latent_q":latent_q}).get("pose")
        return {"pose": pose, "latent_x":latent_x, "latent_q":latent_q}

