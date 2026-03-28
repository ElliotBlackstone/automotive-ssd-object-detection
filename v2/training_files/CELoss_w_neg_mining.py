import torch

def CELoss_w_neg_mining(conf_all: torch.Tensor,
                        cls_t: torch.Tensor,
                        pos_mask: torch.Tensor,
                        neg_pos_ratio: float = 3.0,
                        empty_image_negatives: int = 3,
                        ) -> torch.Tensor:
    """
    SSD classification loss with hard negative mining.

    Inputs
    conf_all:
        Tensor of shape [B, P, C] containing class logits for each prior.
    cls_t:
        Tensor of shape [B, P] containing integer class targets.
        Background class is assumed to be 0.
    pos_mask:
        Boolean tensor of shape [B, P]. True means the prior is matched to a GT box.
    neg_pos_ratio:
        Number of negatives kept per positive, per image.
        Must be nonnegative.
    empty_image_negatives:
        Number of negatives to keep for images with zero positives.
        0 means empty images contribute no classification loss,
        even if false positives are present.

    Output
    Scalar tensor:
        (sum of positive CE + sum of selected negative CE) / max(total_positives, 1)
    """
    B, P, C = conf_all.shape

    # ---------- per-prior CE ----------
    ce = torch.nn.functional.cross_entropy(
        conf_all.reshape(-1, C),
        cls_t.reshape(-1),
        reduction="none",
    ).reshape(B, P)  # [B, P]

    # positives always contribute
    ce_pos = ce[pos_mask].sum()

    # ---------- derive counts from pos_mask ----------
    num_pos_per_img = pos_mask.sum(dim=1)              # [B]
    total_pos = num_pos_per_img.sum().clamp_min(1)     # scalar int tensor
    available_neg_per_img = (~pos_mask).sum(dim=1)     # [B]

    # base negative count = floor(R * n_pos), matching old int(...) behavior for n_pos > 0
    num_neg_per_img = torch.floor(
        num_pos_per_img.to(dtype=ce.dtype) * neg_pos_ratio
    ).to(torch.long)

    # explicit policy for empty images
    if empty_image_negatives > 0:
        num_neg_per_img = torch.where(
            num_pos_per_img == 0,
            torch.full_like(num_neg_per_img, empty_image_negatives),
            num_neg_per_img,
        )

    # cannot select more negatives than actually exist
    num_neg_per_img = torch.minimum(num_neg_per_img, available_neg_per_img)

    # ---------- vectorized hard negative mining ----------
    # Build the mining mask without tracking gradients through the ranking logic.
    # Gradients still flow through ce[...] once the mask is applied.
    with torch.no_grad():
        neg_scores = ce.detach().masked_fill(pos_mask, float("-inf"))  # positives excluded

        # Sort negatives from hardest to easiest within each image.
        order = neg_scores.argsort(dim=1, descending=True)  # [B, P]

        # rank[b, j] = rank position of prior j within image b
        rank = torch.empty_like(order)
        rank.scatter_(
            dim=1,
            index=order,
            src=torch.arange(P, device=order.device).view(1, P).expand(B, P),
        )

        neg_mask = rank < num_neg_per_img.unsqueeze(1)
        neg_mask &= ~pos_mask  # explicit safety

    ce_neg = ce[neg_mask].sum()

    return (ce_pos + ce_neg) / total_pos.to(dtype=ce.dtype)