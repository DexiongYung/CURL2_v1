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
    def __init__(
        self, z_dim: int, critic, batch_sz: int, temp: float, use_cos: bool
    ) -> None:
        super(SimCLR, self).__init__()
        self.encoder = critic.encoder
        self.NTXentLoss = NTXentLoss(
            batch_size=batch_sz, temperature=temp, use_cosine_similarity=use_cos
        )
        self.projection_head = SIMCLR_projection_MLP(z_dim=z_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def create_optimizer(self, lr):
        return torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=lr,
        )

    def compute_NTXent_loss(self, z_anchor, z_pos):
        logits = self.compute_logits(z_anchor, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(z_anchor.get_device())
        return self.cross_entropy_loss(logits, labels)

    def encode(self, x):
        return self.projection_head.forward(self.encoder(x))


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
