import torch
import torch.nn as nn
import torch.nn.functional as F

# positional encoding from nerf
def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


class PoseEncoder(nn.Module):

    def __init__(self, encoder_dim, apply_positional_encoding=True,
                 num_encoding_functions=6, shallow_mlp=False):

        super(PoseEncoder, self).__init__()
        self.apply_positional_encoding = apply_positional_encoding
        self.num_encoding_functions = num_encoding_functions
        self.include_input = True
        self.log_sampling = True
        x_dim = 3
        q_dim = 4
        if self.apply_positional_encoding:
            x_dim = x_dim + self.num_encoding_functions * x_dim * 2
            q_dim = q_dim + self.num_encoding_functions * q_dim * 2
        if shallow_mlp:
            self.x_encoder = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(),
                                           nn.Linear(64,encoder_dim))
            self.q_encoder = nn.Sequential(nn.Linear(q_dim, 64), nn.ReLU(),
                                           nn.Linear(64,encoder_dim))
        else:
            self.x_encoder = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(),
                                           nn.Linear(64,128),
                                           nn.ReLU(),
                                           nn.Linear(128,256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim)
                                           )
            self.q_encoder = nn.Sequential(nn.Linear(q_dim, 64), nn.ReLU(),
                                           nn.Linear(64, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim)
                                       )



        self.x_dim = x_dim
        self.q_dim = q_dim
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, pose):
        if self.apply_positional_encoding:
            encoded_x = positional_encoding(pose[:, :3])
            encoded_q = positional_encoding(pose[:, 3:])
        else:
            encoded_x = pose[:, :3]
            encoded_q = pose[:, 3:]

        latent_x = self.x_encoder(encoded_x)
        latent_q = self.q_encoder(encoded_q)
        return latent_x, latent_q

class MultiSCenePoseEncoder(PoseEncoder):
    def __init__(self, encoder_dim,
                 num_encoding_functions=6, shallow_mlp=False):

        super(MultiSCenePoseEncoder, self).__init__(encoder_dim, apply_positional_encoding=True,
                 num_encoding_functions=num_encoding_functions, shallow_mlp=shallow_mlp)

        scene_dim = 1 + self.num_encoding_functions * 2
        input_x_dim = scene_dim + self.x_dim
        input_q_dim = scene_dim + self.q_dim
        if shallow_mlp:
            self.x_encoder = nn.Sequential(nn.Linear(input_x_dim, 64), nn.ReLU(),
                                           nn.Linear(64, encoder_dim))
            self.q_encoder = nn.Sequential(nn.Linear(input_q_dim, 64), nn.ReLU(),
                                           nn.Linear(64, encoder_dim))
        else:
            self.x_encoder = nn.Sequential(nn.Linear(input_x_dim, 64), nn.ReLU(),
                                           nn.Linear(64, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim)
                                           )
            self.q_encoder = nn.Sequential(nn.Linear(input_q_dim, 64), nn.ReLU(),
                                           nn.Linear(64, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim)
                                           )

        self._reset_parameters()

    def forward(self, pose, scene):
        encoded_x = positional_encoding(pose[:, :3])
        encoded_q = positional_encoding(pose[:, 3:])
        encoded_s = positional_encoding(scene)

        latent_x = self.x_encoder(torch.cat((encoded_x, encoded_s), dim=1))
        latent_q = self.q_encoder(torch.cat((encoded_q, encoded_s), dim=1))
        return latent_x, latent_q

