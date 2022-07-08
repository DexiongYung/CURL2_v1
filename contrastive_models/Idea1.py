import torch
import torch.nn as nn
from contrastive_models.BYOL import BYOL_projection_MLP


class Idea1(nn.Module):
    def __init__(self, z_dim, encoder_A, encoder_B, oracle):
        super(Idea1, self).__init__()
        self.encoder = encoder_A
        self.encoder_target = encoder_B
        self.online_projection = BYOL_projection_MLP(z_dim=z_dim)
        self.target_projection = BYOL_projection_MLP(z_dim=z_dim)
        self.online_predict = BYOL_projection_MLP(z_dim=z_dim)
        self.target_predict = BYOL_projection_MLP(z_dim=z_dim)
        self.oracle = oracle

    def encode(self, x, encoder_A=True):
        if encoder_A:
            x_enc = self.encoder(x)
            x_proj = self.online_projection(x_enc)
            x_pred = self.online_predict(x_proj)
        else:
            x_enc = self.encoder_target(x)
            x_proj = self.target_projection(x_enc)
            x_pred = self.target_predict(x_proj)

        return x_pred

    def create_optimizer(self, lr: float):
        return torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.online_projection.parameters())
            + list(self.online_predict.parameters())
            + list(self.encoder_target.parameters())
            + list(self.target_projection.parameters())
            + list(self.target_predict.parameters()),
            lr=lr,
        ), torch.optim.Adam(self.oracle.parameters(), lr=lr)
