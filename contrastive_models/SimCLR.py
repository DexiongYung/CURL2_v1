import torch
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
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def create_optimizer(self, lr):
        return torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=lr,
        )

    def compute_logits(self, anchor, pos):
        logits = nn.functional.cosine_similarity(
            anchor[:, :, None], pos.t()[None, :, :]
        )
        return logits

    def compute_NCE_loss(self, z_anchor, z_pos):
        logits = self.compute_logits(z_anchor, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(z_anchor.get_device())
        return self.cross_entropy_loss(logits, labels)

    def encode(self, x):
        return self.projection_head.forward(self.encoder(x))
