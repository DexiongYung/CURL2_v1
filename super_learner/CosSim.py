import torch.nn.functional as F


def cos_sim(x, y):
    return F.cosine_similarity(x, y).mean()
