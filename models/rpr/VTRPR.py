import torch.nn as nn
import torch
from models.transposenet.MSTransPoseNet import PoseRegressor

class VTRPR(nn.Module):
    # our proposed PAE-based RPR model for implementing iAPR
    def __init__(self, mstransformer, pae, config):
        super().__init__()

        self.rpr_encoder_dim = config.get("hidden_dim")
        rpr_num_heads = config.get("rpr_num_heads")
        rpr_dim_feedforward = config.get("rpr_dim_feedforward")
        rpr_dropout = config.get("rpr_dropout")
        rpr_activation = config.get("rpr_activation")

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.rpr_encoder_dim,
                                                                 nhead=rpr_num_heads,
                                                                 dim_feedforward=rpr_dim_feedforward,
                                                                 dropout=rpr_dropout,
                                                                 activation=rpr_activation)

        self.rpr_transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
                                                               num_layers=config.get("rpr_num_encoder_layers"),
                                                               norm=nn.LayerNorm(self.rpr_encoder_dim))


        self.ln = nn.LayerNorm(self.rpr_encoder_dim)
        self.rel_x_token = nn.Parameter(torch.zeros((1, self.rpr_encoder_dim)), requires_grad=True)
        self.rel_q_token = nn.Parameter(torch.zeros((1, self.rpr_encoder_dim)), requires_grad=True)
        self.rel_regressor_x =  PoseRegressor(self.rpr_encoder_dim, 3)
        self.rel_regressor_q = PoseRegressor(self.rpr_encoder_dim, 4)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

        self.mstransformer = mstransformer
        self.pae = pae

    def forward(self, query_data, neighbor_pose, neighbor_scene):
        # Get latents from ms-transformer and pae

        res_query = self.mstransformer(query_data)
        query_latent_x = res_query.get("latent_x")
        query_latent_q = res_query.get("latent_q")
        neighbor_latent_x, neighbor_latent_q = self.pae(neighbor_pose, neighbor_scene)

        # make into a sequence and append token
        batch_size = neighbor_pose.shape[0]
        rel_x_token = self.rel_x_token.unsqueeze(1).repeat(1, batch_size, 1)
        rel_q_token = self.rel_q_token.unsqueeze(1).repeat(1, batch_size, 1)

        # S x B x D
        src = torch.cat([rel_x_token, rel_q_token, query_latent_x.unsqueeze(0),
                         query_latent_q.unsqueeze(0),
                         neighbor_latent_x.unsqueeze(0),
                         neighbor_latent_q.unsqueeze(0)])

        # regress latent relative with transformer encoder
        rel_latent_x = self.ln(self.rpr_transformer_encoder(src)[0])
        rel_latent_q = self.ln(self.rpr_transformer_encoder(src)[1])

        # regress relative x and q
        rel_x = self.rel_regressor_x(rel_latent_x)
        rel_q = self.rel_regressor_q(rel_latent_q)

        return {"rel_pose":torch.cat((rel_x,rel_q), dim=1)}






