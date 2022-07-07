import torch
import torch.nn as nn


class CURL2_projection_MLP(nn.Module):
    def __init__(self, z_dim) -> None:
        super(CURL2_projection_MLP, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim)
        )

    def forward(self, x):
        return self.net(x)


class CURL2(nn.Module):
    """
    CURL2
    """

    def __init__(
        self,
        z_dim,
        batch_size,
        critic,
        critic_target,
        output_type="continuous",
    ):
        super(CURL2, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.online_encoder = CURL2_projection_MLP(z_dim=z_dim)
        self.target_encoder = CURL2_projection_MLP(z_dim=z_dim)
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
                z_out = self.target_encoder(z_out)
        else:
            z_out = self.encoder(x)
            z_out = self.online_encoder(z_out)

        if detach:
            z_out = z_out.detach()
        return z_out

    # def update_target(self):
    #    utils.soft_update_params(self.encoder, self.encoder_target, 0.05)

    def compute_logits(self, z_a, z_pos, use_other_env=False, z_other_env=None):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)

        if use_other_env:
            W_oe = torch.matmul(self.W, z_other_env.T)
            new_logits = torch.matmul(z_a, W_oe)

            for i in range(logits.shape[0]):
                new_logits[i, i] = logits[i, i]

            logits = new_logits - torch.max(new_logits, 1)[0][:, None]
        else:
            logits = logits - torch.max(logits, 1)[0][:, None]

        return logits

    def create_optimizer(self, lr):
        return torch.optim.Adam(
            list(self.encoder.parameters())
            + [self.W]
            + list(self.online_encoder.parameters()),
            lr=lr,
        )
