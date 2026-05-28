from scipy.optimize import linear_sum_assignment  # CPU (M is tiny)
import numpy as np
import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)


def hungarian_per_sample(
    logits: torch.Tensor,     # (N, M)
    labels: torch.Tensor,     # (N)
    valid_gt: torch.Tensor    # (M)
) -> torch.LongTensor:        # length-M permutation
    device = logits.device
    M = logits.size(1)

    # hard prediction for every pixel
    preds = logits.argmax(-1)                          # (N)

    # pre-compute area of every predicted cluster
    pred_sizes = torch.bincount(preds, minlength=M).float().to(device)

    # initialise the cost matrix
    cost = torch.full((M, M), 0.0, device=device)      # float

    for g in torch.nonzero(valid_gt, as_tuple=False).flatten():
        mask_g = labels.eq(g)
        if mask_g.any():
            gt_size = mask_g.sum().float()             # |g|
            # intersection with every predicted cluster
            inter = torch.bincount(preds[mask_g], minlength=M).float()
            # union = |g| + |pred_m| – intersection
            union = gt_size + pred_sizes - inter
            iou = inter / union.clamp(min=1.0)         # avoid divide-by-zero
            cost[g] = -iou                             # larger IoU → lower cost

    # Hungarian assignment on CPU
    row, col = linear_sum_assignment(cost.cpu().numpy())
    perm = torch.arange(M)
    perm[row] = torch.tensor(col)
    return perm.to(device)


def hungarian_per_batch(
    logits: torch.Tensor,     # (B, N, M)
    labels: torch.Tensor,     # (B, N)  values in [0, M-1] or `ignore_index` (negative)
    valid: torch.Tensor,      # (B, M)  bool
    ignore_index: int = -1,
) -> torch.LongTensor:        # (B, M) permutations on logits.device
    """Vectorised batched IoU-cost Hungarian matching.

    Equivalent to looping `hungarian_per_sample` over the batch dim, but
    builds all (B, M, M) cost matrices on GPU and performs a single CPU
    transfer. The original per-sample variant calls `cost.cpu().numpy()`
    inside the loop, which forces one GPU->CPU sync per sample and per
    loss term — at batch_size=64 with low+high res, that is 128 syncs per
    step and was the dominant cost in profiling.
    """
    B, N, M = logits.shape
    device = logits.device

    # `valid` may have width K <= M (the per-sample variant only used it to
    # iterate non-zero indices, so callers pass the GT-objects mask whose
    # width is num_objects, not num_traj). Pad/truncate to width M so the
    # broadcast against the (B, M, M) cost matrix is well-defined.
    K = valid.shape[1]
    if K < M:
        pad = torch.zeros(B, M - K, dtype=valid.dtype, device=valid.device)
        valid = torch.cat([valid, pad], dim=1)
    elif K > M:
        valid = valid[:, :M]

    preds = logits.argmax(-1)                                              # (B, N)

    # mask out invalid pixels (label < 0 / == ignore_index). Also clamp safe_labels
    # to [0, M-1] in case of unexpected out-of-range values; valid_pix gates them out.
    valid_pix = (labels >= 0) & (labels < M)                               # (B, N)
    safe_labels = torch.where(valid_pix, labels, torch.zeros_like(labels)).long()
    safe_preds = preds.long()

    # joint count: inter[b, g, p] = #{ pixels n: labels[b,n]==g & preds[b,n]==p & valid_pix }
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    flat_idx = (batch_idx * (M * M) + safe_labels * M + safe_preds).reshape(-1)
    contrib = valid_pix.float().reshape(-1)
    inter_flat = torch.zeros(B * M * M, device=device, dtype=torch.float)
    inter_flat.scatter_add_(0, flat_idx, contrib)
    inter = inter_flat.view(B, M, M)                                       # (B, gt, pred)

    # gt_sizes[b, g]   = #{ pixels with label g }            -> sum over preds
    # pred_sizes[b, p] = #{ pixels predicted as p }          -> over ALL pixels (matches original)
    gt_sizes = inter.sum(dim=2)                                            # (B, M)
    pred_idx_flat = (batch_idx * M + safe_preds).reshape(-1)
    pred_flat = torch.zeros(B * M, device=device, dtype=torch.float)
    pred_flat.scatter_add_(0, pred_idx_flat, torch.ones_like(pred_idx_flat, dtype=torch.float))
    pred_sizes = pred_flat.view(B, M)                                      # (B, M)

    union = gt_sizes.unsqueeze(2) + pred_sizes.unsqueeze(1) - inter        # (B, M, M)
    iou = inter / union.clamp(min=1.0)
    cost = -iou

    # zero out rows for invalid GTs so they don't bias the assignment
    cost = cost * valid.to(cost.dtype).unsqueeze(2)

    # ONE GPU->CPU sync for the whole batch
    cost_np = cost.detach().cpu().numpy()                                  # (B, M, M)
    perms = np.tile(np.arange(M, dtype=np.int64), (B, 1))                  # (B, M)
    for b in range(B):
        row, col = linear_sum_assignment(cost_np[b])
        perms[b, row] = col
    return torch.from_numpy(perms).to(device)






def instance_balanced_ce(logits_aligned, labels, ignore_index):
    B, N, M = logits_aligned.shape
    logits = logits_aligned.transpose(1, 2)          # [B, M, N]

    # 1) ordinary CE per pixel, no reduction
    ce = F.cross_entropy(logits, labels,
                         ignore_index=ignore_index,
                         reduction='none')            # [B, N]

    # 2) compute the area (number of pixels) of every GT instance in each batch item
    #    area[b, g] = |{ n : labels[b,n] == g }|
    areas = torch.zeros(B, M, device=labels.device, dtype=torch.float)
    for b in range(B):
        valid = labels[b] != ignore_index
        if valid.any():
            counts = torch.bincount(labels[b, valid], minlength=M).float()
            areas[b] = counts + 1e-6                 # avoid division by zero

    # 3) weight for each instance g is 1 / area[b,g]
    weights_per_inst = 1.0 / areas                   # [B, M]

    # 4) build a per-pixel weight map w_map[b,n] = weight of the instance that pixel belongs to
    #    gather needs labels >= 0, so clamp negative ignore_index to 0 first
    indexable = labels.clamp(min=0)
    w_map = weights_per_inst.gather(1, indexable)    # [B, N]
    w_map[labels == ignore_index] = 0.0              # ignore void pixels

    # 5) final instance-balanced cross-entropy
    loss = (ce * w_map).sum() / w_map.sum()
    return loss




def focal_loss(
    logits_aligned: torch.Tensor,  # shape [B, N, M]
    labels: torch.Tensor,          # shape [B, N]
    ignore_index: int = -100,
    gamma: float = 2.0,            # focusing parameter
    alpha=None,                    # scalar or 1-D tensor of length M
    reduction: str = "mean"        # "mean", "sum", or "none"
) -> torch.Tensor:
    """
    Multi-class focal loss for per-pixel segmentation.

    logits_aligned : raw scores, one row per pixel (B, N, M)
    labels         : integer GT ids in 0 … M-1 or ignore_index  (B, N)
    """
    B, N, M = logits_aligned.shape
    # flatten spatial/temporal dimension so masking is easy
    logits_flat = logits_aligned.reshape(-1, M)   # [(B·N), M]
    labels_flat = labels.reshape(-1)              # [(B·N)]

    # keep only valid pixels
    valid = labels_flat != ignore_index
    if valid.sum() == 0:
        # nothing to compute
        return torch.zeros((), device=logits_aligned.device,
                           dtype=logits_aligned.dtype, requires_grad=True)

    logits_flat = logits_flat[valid]              # [P, M]
    labels_flat = labels_flat[valid]              # [P]

    # log-probs and probs of the true class
    log_probs = F.log_softmax(logits_flat, dim=-1)      # [P, M]
    probs     = log_probs.exp()                         # [P, M]
    log_p_t   = log_probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [P]
    p_t       = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)      # [P]

    # class-balancing factor α_t
    if alpha is None:
        alpha_t = 1.0
    elif isinstance(alpha, (float, int)):
        alpha_t = float(alpha)
    else:
        alpha_vec = torch.as_tensor(alpha, device=logits_aligned.device,
                                    dtype=logits_aligned.dtype)
        alpha_t = alpha_vec[labels_flat]           # [P]

    # focal loss computation
    loss = -alpha_t * (1.0 - p_t).pow(gamma) * log_p_t  # [P]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:  # "none"
        # reshape back to (B, N) order of original pixels
        out = torch.zeros(B * N, device=logits_aligned.device,
                          dtype=loss.dtype)
        out[valid] = loss
        return out.view(B, N)
    
    
    
    
    

def dice_loss(
    logits_aligned: torch.Tensor,   # [B, N, M]  (B=batch, N=pixels/tokens, M=classes or masks)
    labels: torch.Tensor,           # [B, N]     integer IDs 0‥M-1 or ignore_index
    ignore_index: int = -100,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Combined loss = ce_weight * CrossEntropy + dice_weight * Dice.
    Dice is computed per class (or per mask channel) and averaged
    over the classes that appear in the ground truth for this batch.
    """
    # ---------- Dice part ----------
    B, N, M = logits_aligned.shape

    # probabilities for each class
    probs = F.softmax(logits_aligned, dim=-1)         # [B, N, M]

    # flatten to 1-D over pixels so we can mask invalid ones
    probs_flat  = probs.reshape(-1, M)                # [(B·N), M]
    labels_flat = labels.reshape(-1)                  # [(B·N)]

    valid = labels_flat != ignore_index
    if valid.sum() == 0:
        # no valid pixels – return CE term only
        return 0

    probs_flat  = probs_flat[valid]                   # [P, M]
    labels_flat = labels_flat[valid]                  # [P]

    # one-hot target mask
    gt_onehot = F.one_hot(labels_flat, num_classes=M).float()  # [P, M]

    # per-class intersection and union
    intersection = (probs_flat * gt_onehot).sum(dim=0)         # [M]
    union        = probs_flat.sum(dim=0) + gt_onehot.sum(dim=0)  # [M]

    dice_per_class = (2 * intersection + eps) / (union + eps)  # [M]

    # restrict Dice average to classes that actually appear
    present = gt_onehot.sum(dim=0) > 0                         # [M] bool
    dice_loss = 1.0 - dice_per_class[present].mean()

    # ---------- Combine ----------
    return dice_loss