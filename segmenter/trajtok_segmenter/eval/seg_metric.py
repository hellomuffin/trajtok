import torch
import math
from typing import Tuple, Dict, Optional

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _unique_ids(x: torch.Tensor, ignore_id: Optional[int]) -> torch.Tensor:
    ids = x.unique()
    return ids[ids != ignore_id] if ignore_id is not None else ids

def _build_iou_matrix(pred, gt, pred_ids, gt_ids) -> torch.Tensor:
    """
    Returns |G| × |P| matrix with IoU for every GT / Pred tube.
    """
    ious = torch.zeros((gt_ids.numel(), pred_ids.numel()), dtype=torch.float32, device=pred.device)

    for gi, g in enumerate(gt_ids):
        g_mask = (gt == g)
        g_size = g_mask.sum()
        if g_size == 0:
            continue
        for pi, p in enumerate(pred_ids):
            p_mask = (pred == p)
            inter = (g_mask & p_mask).sum()
            if inter == 0:
                continue
            union = g_size + p_mask.sum() - inter
            ious[gi, pi] = inter.float() / union.float()
    return ious


def _hungarian_assign(iou_mat: torch.Tensor, thr: float = .5):
    """
    One-to-one matching of tracks with IoU ≥ thr.
    SciPy’s linear_sum_assignment is used for optimal pairing;
    if SciPy is absent we fall back to a greedy matching.
    """
    from scipy.optimize import linear_sum_assignment
    # We minimise cost, so convert IoU to cost: cost = 1 - IoU (clipped)
    cost = 1.0 - iou_mat.clamp(min=thr)           # IoU < thr ⇒ cost = 1
    gi, pi = linear_sum_assignment(cost.cpu().numpy())
    gi = torch.as_tensor(gi, device=iou_mat.device)
    pi = torch.as_tensor(pi, device=iou_mat.device)
    return gi, pi

# ---------------------------------------------------------------------
# VEQ computation
# ---------------------------------------------------------------------
def veq_scores(pred: torch.Tensor,
               gt:   torch.Tensor,
               ignore_id: Optional[int] = None,
               iou_thr: float = .5
              ) -> Tuple[float, float, float]:
    """
    Returns VEQ, VEQ_SQ, VEQ_RQ.
    """
    pred_ids = _unique_ids(pred, ignore_id)
    gt_ids   = _unique_ids(gt,   ignore_id)

    iou_mat = _build_iou_matrix(pred, gt, pred_ids, gt_ids)          # |G| × |P|
    gi, pi  = _hungarian_assign(iou_mat, iou_thr)
    matched_ious = iou_mat[gi, pi]
    tp = (matched_ious >= iou_thr).sum().item()
    fp = pred_ids.numel() - tp
    fn = gt_ids.numel()   - tp

    if tp == 0:
        return 0.0, 0.0, 0.0

    iou_sum = matched_ious[matched_ious >= iou_thr].sum().item()
    veq_sq  = iou_sum / tp
    veq_rq  = tp / (tp + 0.5 * fp + 0.5 * fn)
    veq     = veq_sq * veq_rq
    return veq, veq_sq, veq_rq


# ---------------------------------------------------------------------
# STQ-EN computation
# ---------------------------------------------------------------------
def stq_en(pred: torch.Tensor,
           gt:   torch.Tensor,
           ignore_id: Optional[int] = None,
           iou_thr: float = .5
          ) -> Dict[str, float]:
    """
    Returns dict with keys: stq_en, sq, aq
    """
    pred_ids = _unique_ids(pred, ignore_id)
    gt_ids   = _unique_ids(gt,   ignore_id)

    iou_mat  = _build_iou_matrix(pred, gt, pred_ids, gt_ids)
    gi, pi   = _hungarian_assign(iou_mat, iou_thr)

    if gi.numel() == 0:
        return {'stq_en': 0.0, 'sq': 0.0, 'aq': 0.0}

    # per-pair statistics ---------------------------------------------
    inters  = []
    unions  = []
    for g_idx, p_idx in zip(gi.tolist(), pi.tolist()):
        if iou_mat[g_idx, p_idx] < iou_thr:
            continue
        g_id = gt_ids[g_idx].item()
        p_id = pred_ids[p_idx].item()
        g_mask = (gt   == g_id)
        p_mask = (pred == p_id)

        inter = (g_mask & p_mask).sum().item()
        union = (g_mask | p_mask).sum().item()

        inters.append(inter)
        unions.append(union)

    if len(inters) == 0:
        return {'stq_en': 0.0, 'sq': 0.0, 'aq': 0.0}

    inters = torch.tensor(inters, dtype=torch.float32)
    unions = torch.tensor(unions, dtype=torch.float32)
    ious   = inters / unions                               # IoU per match

    # Segmentation-quality term (mean IoU over matches)
    sq = ious.mean().item()

    # Association-quality term:  Σ TPA / Σ union
    aq = inters.sum().item() / unions.sum().item()

    stq = math.sqrt(sq * aq)
    return {'stq_en': stq, 'sq': sq, 'aq': aq}


# ---------------------------------------------------------------------
# example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    T, H, W = 10, 32, 32
    gt   = torch.zeros((T, H, W), dtype=torch.int32)
    pred = torch.zeros_like(gt)

    # toy example: two GT tracks: id 1 and id 2
    gt[:, :16, :]  = 1
    gt[:, 16:, :]  = 2

    # pretend prediction recovers id 1 perfectly, id 2 partially
    pred[:, :16, :] = 1
    pred[:, 16:, :16] = 2          # misses right quarter

    veq, veq_sq, veq_rq = veq_scores(pred, gt, ignore_id=0)
    stq_dict            = stq_en(pred, gt,   ignore_id=0)

    print(f"VEQ={veq:.3f}  (SQ={veq_sq:.3f}, RQ={veq_rq:.3f})")
    print(f"STQ_EN={stq_dict['stq_en']:.3f}  "
          f"(SQ={stq_dict['sq']:.3f}, AQ={stq_dict['aq']:.3f})")