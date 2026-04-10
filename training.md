# Training Method and Results

Both version 1 and version 2 models were trained in the same manor.  The only difference is the training time for version 2 models was reduced by half due to many optimizations described later.

## Training methodology

Each training image is passed through a torchvision v2 transform pipeline that converts it to a float32 tensor, then applies a size-aware IoU-based random crop (`ConditionalIoUCrop`) which chooses between two `RandomIoUCrop` policies depending on whether the image contains large or only small objects, so that small targets are more likely to be zoomed in on. This is followed by bounding-box sanitization, random horizontal flips, and random photometric distortion for geometric and color augmentation, before finally resizing to 300×300 and normalizing with ImageNet mean and standard deviation to match the SSD backbone’s expected input distribution.

The SSD detector is trained end-to-end with a standard localization + classification objective and hard negative mining. For each batch, the model predicts localization offsets and class logits for all 8,732 priors; these are matched to ground-truth boxes using an IoU-style overlap, and only matched priors are treated as positives. The localization loss is Smooth L1 on the encoded offsets for positive priors, normalized by the number of positives. The classification loss is a cross-entropy over all classes (including background), but it is computed on all positive priors plus a subset of the hardest negatives: for each image, negative priors are ranked by their per-prior CE loss and only the top-k are kept, with a configurable negative-to-positive ratio (e.g. 3:1). This focuses learning on informative background examples and prevents easy negatives from dominating the gradients. During evaluation, the same target-building and loss computation are reused, and detection quality is measured via torchmetrics `MeanAveragePrecision` at IoU=0.5 (mAP@0.50), using the model’s own `predict` method for decoding and NMS. 

Optimization uses SGD with Nesterov momentum and weight decay, together with a cosine learning-rate schedule with linear warmup, see learning-rate vs. epoch plot below for an example with 150 epochs (which was used for model training).

![Cosine LR scheudle](/figures/LR_plot.png)

The helper `build_optimizer_and_scheduler` builds an SGD optimizer (chosen values: base LR 3e-3, momentum 0.9, weight decay 5e-4) and a `LambdaLR` scheduler that first linearly increases the learning rate from 0 to the base LR over a specified number of warmup epochs, then decays it following a cosine curve down to a minimum LR (chosen value: 1e-6) over the remaining training steps. The scheduler is designed to be stepped once per optimizer step (per mini-batch), and the training loop optionally supports stepping per batch or per epoch depending on the `sched_step_w_opt` flag. The trainer also supports early stopping based on validation mAP, periodic and “best” checkpoint saving (including optimizer/scheduler state and RNG state), and utility functions to merge and plot loss/mAP curves across multiple training runs.

Models were trained for 150 epochs with the previously described optimizer and schuduler using [SSD_model_train.ipynb](/v1/SSD_model_train.ipynb) for version 1 models and [train_model.py](/v2/training_files/train_model.py) for version 2 models.


## Results (v1)
The first model (named "Zoom out, no bootstrap") was trained with an additional image transformation, `RandomZoomOut`.  The second model (named "No zoom out, no bootstrap") was trained with no additional augmentations.  The final model (named "No zoom out, bootstrap") had the training set “bootstrapped” by oversampling image filenames according to how many objects they contain: images with 0 objects are used once, those with 1–2 objects are duplicated, those with 3–6, 7–9, and ≥10 objects are repeated 3, 4, and 5 times respectively in the `file_list`. This weighted duplication increases the effective number of training samples and biases each epoch toward images with richer annotations, without fabricating any synthetic labels.  The training loss/mAP@0.50 information per epoch for the top performing model is below.  The mAP subplot is mAP@0.50 evaluated on the validation set per epoch.

![Training loss data](/figures/loss_vs_epoch.png)

The mAP@0.50, along with individual class mAP@0.50, on the test set are reported in the table below.

| Model    | mAP@0.50 | biker | car | pedestrian | traffic light | truck |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| Zoom out, no bootstrap     | 0.4613     | 0.2854     | 0.6618     | 0.2145     | 0.5571     | 0.5875     |
| No zoom out, no bootstrap  | 0.4724     | 0.3015     | 0.6681     | 0.2314     | 0.5619     | 0.5992     |
| No zoom out, bootstrap     | **0.5292** | **0.4045** | **0.7088** | **0.2907** | **0.5956** | **0.6465** |

Since the vast majority of objects in the dataset are small compared to the image size, it is not a surprise that the `RandomZoomOut` transformation degrades model performance.  Enlarging the training dataset via the "bootstrapping" method significantly improved model performance.  Training time for each model was ~50 hours.


## Upgrades to the v2 models

The second version of the SSD model includes several structural and performance-focused improvements identified through profiling of the training and evaluation pipeline. Profiling exposed weak links in the original implementation, which motivated a cleaner file structure, the use of `torch.autocast` during the forward pass to reduce the cost of one of the most expensive stages of training and testing, and a batched implementation of [`build_targets`](/v2/training_files/build_targets.py) to remove per-image bottlenecks. The updated version also adds batched NMS variants through the [`gen-nms-package`](https://github.com/ElliotBlackstone/gen-nms-package), further improving postprocessing efficiency, and uses better DataLoader settings selected from benchmarking in ['sweep_dl_configs'](/benchmarking/sweep_dl_configs.py). Together, these changes make the v2 pipeline more organized, more efficient, and better suited for larger-scale training and evaluation.  In total, training time was reduced from ~50 hours to ~25 hours.


### Training setup for IoU-variant models

The following four SSD variants were trained and evaluated using their matching overlap metric:

- **IoU**
- **GIoU**
- **DIoU**
- **CIoU**

Each model used the same general detection and hard-negative mining setup, with the main differences being the selected IoU variant and the `iou_thresh` value.

#### Shared settings

| Parameter | Value |
| :-- | :--: |
| `neg_pos_ratio` | `3.0` |
| `score_thresh` | `0.2` |
| `nms_thresh` | `0.3` |
| `max_detections_per_img` | `200` |

#### Per-model configuration

| Model | IoU variant used | `iou_thresh` |
| :--: | :--: | :--: |
| IoU  | IoU  | `0.50` |
| GIoU | GIoU | `0.45` |
| DIoU | DIoU | `0.45` |
| CIoU | CIoU | `0.40` |

These experiments were designed to compare how different IoU-based matching/objective variants affect SSD detection performance under otherwise consistent training and inference settings.  For comparison, the v1 models used the same shared settings but with `max_detections_per_img` lowered to `100` and used `iou_thresh` of `0.40`.  As for IoU variants, the v1 models used CIoU within the `build_targets` function and DIoU for NMS.


## Results (v2)

As previously mentioned the training time of the v2 models was reduced from ~50 hours to ~25 hours.  The mAP@0.50 on the validation set during training is displayed below.  The top performing v1 model is also shown for comparison.

![mAP comparison plot](/figures/mAP_on_val.png)

The mAP@0.50, along with individual class mAP@0.50, on the test set are reported in the table below.

| Model | mAP@0.50     |  biker     |   car      | pedestrian     | traffic light     |  truck     |
| :---: | :------:     | :----:     | :----:     | :--------:     | :-----------:     | :----:     |
|  IoU  |  0.4022      | 0.2815     | 0.5529     |   0.1958       |     0.4222        | 0.5585     |
|  GIoU |  0.3971      | 0.2630     | 0.5438     |   0.1937       |     0.4152        | 0.5697     |
|  DIoU |  **0.5383**  | **0.4048** | **0.7222** |   **0.3052**   |     **0.6011**    | **0.6582** |
|  CIoU |  0.5199      | 0.3980     | 0.7002     |   0.2881       |     0.5933        | 0.6201     |
|  v1   |  0.5292      | 0.4045     | 0.7088     |   0.2907       |     0.5956        | 0.6465     |


