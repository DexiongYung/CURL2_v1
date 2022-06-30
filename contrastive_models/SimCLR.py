import torch.nn as nn


class SIMCLR_projection_MLP(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(SIMCLR_projection_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, z_dim: int, critic) -> None:
        super(SimCLR, self).__init__()
        self.encoder = critic.encoder
        self.projection_head = SIMCLR_projection_MLP(z_dim=z_dim)

    def encode(self, x):
        return self.projection_head.forward(self.encoder(x))

    def compute_logits(self, anchor, pos):
        logits = nn.functional.cosine_similarity(
            anchor[:, :, None], pos.t()[None, :, :]
        )
        return logits
