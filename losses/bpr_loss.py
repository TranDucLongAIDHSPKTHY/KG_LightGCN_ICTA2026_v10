"""losses/bpr_loss.py — BPR loss (Rendle et al., UAI 2009) — v10.

CHANGELOG v10-fix:
  - kg_bpr_loss: SỬA DẤU SAI — pos_score là TransR distance (nhỏ = tốt),
    neg_score là distance của negative triple.
    Paper KGAT: maximize log σ(neg_score - pos_score)
    → loss = -log σ(neg_score - pos_score) = -logsigmoid(neg_score - pos_score)
    Code cũ có: -logsigmoid(neg_scores - pos_scores) ← ĐÚNG về mặt ký hiệu
    NHƯNG trong _kgat_step: pos_score, neg_score được gọi theo thứ tự (h,r,tp,tn)
    và kg_bpr_loss(pos_score, neg_score) → cần verify chiều gọi là đúng.

  Để tránh nhầm lẫn: đặt tham số rõ ràng là
    kg_bpr_loss(pos_dist, neg_dist)
    với pos_dist = ||h+r-t_pos||^2 (nhỏ hơn = triple đúng)
         neg_dist = ||h+r-t_neg||^2 (lớn hơn = triple sai)
    Loss = -logsigmoid(neg_dist - pos_dist)  ← muốn neg > pos
"""
import torch
import torch.nn.functional as F


def bpr_loss(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> torch.Tensor:
    """
    BPR loss cho CF recommendation.
    Loss = -mean log σ(score_pos - score_neg)
    """
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def kg_bpr_loss(
    pos_dist: torch.Tensor,
    neg_dist: torch.Tensor,
) -> torch.Tensor:
    """
    BPR-style TransR loss cho KG triple scoring (KGAT, KDD 2019).

    Args:
        pos_dist: ||h + r - t_pos||^2  — khoảng cách của positive triple
                  (nhỏ hơn nghĩa là triple đúng hơn)
        neg_dist: ||h + r - t_neg||^2  — khoảng cách của negative triple
                  (lớn hơn nghĩa là triple sai)

    Loss = -mean log σ(neg_dist - pos_dist)
    Mục tiêu: neg_dist >> pos_dist (negative triple cách xa hơn positive)

    LƯU Ý: v10-fix — trước đây code gọi kg_bpr_loss(pos_score, neg_score)
    từ _kgat_step nhưng pos_score/neg_score là OUTPUT từ kgat.kg_forward()
    vốn đã là squared distances. Chiều gọi (pos, neg) → loss(neg-pos) là ĐÚNG.
    """
    return -F.logsigmoid(neg_dist - pos_dist).mean()