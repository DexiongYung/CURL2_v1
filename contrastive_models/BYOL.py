import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOL_projection_MLP(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(BYOL_projection_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(
        self,
        z_dim,
        critic,
        critic_target,
    ):
        super(BYOL, self).__init__()
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.online_projection = BYOL_projection_MLP(z_dim=z_dim)
        self.target_projection = BYOL_projection_MLP(z_dim=z_dim)
        self.online_predict = BYOL_projection_MLP(z_dim=z_dim)

    def encode(self, x, detach=False, target=False):
        if target:
            with torch.no_grad():
                z_out = self.encoder_target(x)
                z_fin = self.target_projection(z_out)
        else:
            z_out = self.encoder(x)
            z_proj = self.online_projection(z_out)
            z_fin = self.online_predict(z_proj)

        if detach:
            z_fin = z_fin.detach()
        return z_fin

    def compute_L2_MSE(self, z_a, z_pos):
        z_a_norm = F.normalize(z_a, dim=-1, p=2)
        z_pos_norm = F.normalize(z_pos, dim=-1, p=2)
        return 2 - 2 * (z_a_norm * z_pos_norm).sum(dim=-1)
