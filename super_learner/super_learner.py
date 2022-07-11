import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from curl_sac import Actor, gaussian_logprob, squash


class SuperLeaner(nn.Module):
    def __init__(self, pretrained_actor: Actor) -> None:
        super().__init__()
        self.pretrained_actor = pretrained_actor
        self.aug_encoder = copy.deepcopy(self.pretrained_actor.encoder)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = self.pretrained_actor.encoder(obs, detach=detach_encoder)

        mu, log_std = self.pretrained_actor.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        self.outputs["mu"] = mu
        self.outputs["std"] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std
