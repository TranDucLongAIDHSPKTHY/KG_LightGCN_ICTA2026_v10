"""losses/bpr_loss.py — BPR loss (Rendle et al., UAI 2009) — v10."""
import torch
import torch.nn.functional as F


def bpr_loss(user_emb: torch.Tensor, pos_emb: torch.Tensor,
             neg_emb: torch.Tensor) -> torch.Tensor:
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def kg_bpr_loss(pos_scores: torch.Tensor,
                neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(neg_scores - pos_scores).mean()
