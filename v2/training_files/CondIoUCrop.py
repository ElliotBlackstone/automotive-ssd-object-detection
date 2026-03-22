import torch
from torchvision.transforms import v2


class ConditionalIoUCrop(torch.nn.Module):
    """
    Size-aware IoU-based random cropping for object detection.

    This module wraps two `torchvision.transforms.v2.RandomIoUCrop` instances and
    chooses between them based on the relative area of the ground-truth boxes
    in the current image.

    The logic is:

        - Compute the area fraction of each box: area / (H * W).
        - If at least one box has area fraction >= `min_area_frac`
          ("large" object present), use `iou_crop_large`.
        - Otherwise (all boxes are small), use `iou_crop_small`, which is
          typically configured to zoom in more aggressively and/or use looser
          IoU constraints.

    This is intended for datasets with many small objects, where standard
    IoU-based cropping often fails to generate useful zoomed-in views of
    small targets.

    Parameters
    ----------
    min_area_frac : float, optional (default: 0.002)
        Threshold on box area fraction (box_area / (H * W)) used to decide
        whether the image contains any "large" objects. If any box has
        area_frac >= min_area_frac, the "large-object" crop policy is used;
        otherwise the "small-object" crop policy is used.

    small_min_scale : float, optional (default: 0.3)
        `min_scale` passed to the small-object `RandomIoUCrop`. Controls the
        minimum relative size of the sampled crop when all boxes are small.
        Smaller values allow tighter crops (stronger zoom-in).

    large_min_scale : float, optional (default: 0.6)
        `min_scale` passed to the large-object `RandomIoUCrop`. Controls the
        minimum relative size of the sampled crop when at least one box is
        considered large.

    max_scale : float, optional (default: 1.0)
        `max_scale` passed to both `RandomIoUCrop` instances. Controls the
        maximum relative size of the sampled crop.

    min_aspect_ratio : float, optional (default: 0.75)
        Lower bound on the aspect ratio (w / h) for sampled crops, passed to
        both `RandomIoUCrop` instances.

    max_aspect_ratio : float, optional (default: 1.33)
        Upper bound on the aspect ratio (w / h) for sampled crops, passed to
        both `RandomIoUCrop` instances.

    small_sampler_options : sequence of float or None, optional
        `sampler_options` for the small-object `RandomIoUCrop`. Each entry is
        interpreted as a minimum IoU threshold that the sampled crop must
        satisfy with at least one box (or `None` to allow unconstrained crops).
        Including low values and `None` is useful when all boxes are tiny.

    large_sampler_options : sequence of float or None, optional
        `sampler_options` for the large-object `RandomIoUCrop`, used when at
        least one box is larger than `min_area_frac`.

    trials : int, optional (default: 10)
        Number of attempts for each `RandomIoUCrop` to find a valid crop
        before falling back to returning the original image and boxes.

    Forward
    -------
    forward(img, target) -> Tuple[Tensor, Dict[str, Any]]

    Parameters
    ----------
    img : Tensor
        Image tensor of shape (C, H, W).

    target : dict
        Target dictionary expected to contain the key `"boxes"` with a
        tensor of shape (N, 4) in (x1, y1, x2, y2) format, in the same
        coordinate system as `img`. If `"boxes"` is missing or empty,
        the transform returns `(img, target)` unchanged.

    Returns
    -------
    img_out : Tensor
        Cropped (or original) image tensor.

    target_out : dict
        Updated target dictionary with boxes and any other box-dependent
        fields transformed consistently with the image. The exact behavior
        follows `torchvision.transforms.v2.RandomIoUCrop`.

    Notes
    -----
    - This transform does not change images with no boxes.
    - When all objects are small (no box exceeds `min_area_frac`), the
      "small-object" crop policy is applied explicitly instead of skipping
      cropping. This is intended to produce more training examples where
      small targets occupy a larger fraction of the crop.
    """
    def __init__(
        self,
        *,
        min_area_frac: float = 0.02,          # threshold separating "has big box" vs "all small"
        small_min_scale: float = 0.3,         # more aggressive zoom-in for small-only images
        large_min_scale: float = 0.6,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.75,
        max_aspect_ratio: float = 1.33,
        small_sampler_options = (0.0, 0.05, 0.1, 2.0),
        large_sampler_options = (0.05, 0.1, 0.3, 2.0),
        trials: int = 10,
    ):
        super().__init__()
        self.min_area_frac = float(min_area_frac)

        # crop mode when at least one reasonably large box exists
        self.iou_crop_large = v2.RandomIoUCrop(
            min_scale=large_min_scale,
            max_scale=max_scale,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            sampler_options=list(large_sampler_options),
            trials=trials,
        )

        # crop mode when all boxes are small: zoom in more, looser IoU constraints
        self.iou_crop_small = v2.RandomIoUCrop(
            min_scale=small_min_scale,
            max_scale=max_scale,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            sampler_options=list(small_sampler_options),
            trials=trials,
        )

    @torch.no_grad()
    def forward(self, img, target):
        H, W = img.shape[-2], img.shape[-1]
        boxes = target.get("boxes", None)
        if boxes is None or boxes.numel() == 0:
            return img, target

        b = torch.as_tensor(boxes)
        area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
        area_frac = area / float(H * W)

        has_large = (area_frac >= self.min_area_frac).any()

        if has_large:
            img_out, tgt_out = self.iou_crop_large(img, target)
        else:
            # all boxes small → aggressively zoom-in around them
            img_out, tgt_out = self.iou_crop_small(img, target)

        return img_out, tgt_out