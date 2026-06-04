"""losses/contrastive_loss.py — InfoNCE / NT-Xent contrastive loss — v10."""
import torch
import torch.nn.functional as F


def infonce_loss(view1: torch.Tensor, view2: torch.Tensor,
                 temperature: float = 0.2) -> torch.Tensor:
    v1 = F.normalize(view1, dim=-1)
    v2 = F.normalize(view2, dim=-1)
    sim = torch.matmul(v1, v2.T) / temperature
    labels = torch.arange(len(v1), device=v1.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def ssl_loss(emb_aug1: torch.Tensor, emb_aug2: torch.Tensor,
             temperature: float = 0.2) -> torch.Tensor:
    return infonce_loss(emb_aug1, emb_aug2, temperature)
