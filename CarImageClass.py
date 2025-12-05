import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
import pathlib
from PIL import Image
from typing import Tuple, Dict, Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedGroupKFold
import warnings


class ImageClass(Dataset):
    """
    A Dataset class responsible for reading in images with corresponding bounding box and classification labels.

    Inputs
    targ_dir: Directory location of images and .csv file which
              contains bounding box and classification information.
              There should only be on .csv file in targ_dir.
    file_list: List of file names within targ_dir to work with.  Default None, i.e. use all files.
    file_pct: Float between 0 and 1 used to randomly select a percentage of total files to work with.
              Default is 1, i.e. use all files.
    rand_seed: Random seed for reproducability used when file_pct is less than 1.  Default is 724.
    """
    def __init__(self,
                 targ_dir: str,
                 file_list: list | None = None,
                 transform = None,
                 file_pct: float = 1,
                 rand_seed: int | None = 724,
                 include_area: bool = False,
                 ) -> None:
        
        # Create class attributes
        self.directory = targ_dir
        self.transform = transform
        self.paths, self.annotate_df = get_file_path_plus_dataframe(targ_dir=targ_dir, rand_seed=rand_seed, file_list=file_list, file_pct=file_pct)
        self.classes = list(self.annotate_df['class'].unique())
        if 'empty' in self.classes:
            self.classes.remove('empty')
        self.classes.sort() # alphabetical sort of classes
        self.class_to_idx = dict(zip(self.classes, range(0, len(self.classes))))
        self.area = include_area

        self.annotate_df_mapped = self.annotate_df.copy()
        self.annotate_df_mapped["class"] = self.annotate_df_mapped["class"].map(self.class_to_idx)

        self._by_file = {fname: g.reset_index(drop=True) for fname, g in self.annotate_df_mapped.groupby("filename")}
    

    # Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns img, target where:

        img: tv_tensors.Image or torch.Tensor [C,H,W], float32
        target = {
            "boxes":    Tensor[n_i, 4]  # float32, xyxy, absolute pixels
            "labels":   Tensor[n_i]     # int64, in {0, ..., num_classes-1}
            "image_id": Tensor[1]       # int64 (index)
            "areas":  Tensor[n_i]       # float32, added if include_area=True
            }
        """
        # 1) Load image
        img_path = str(self.paths[index])
        img = decode_image(img_path)  # expects CHW
        _, H, W = img.shape

        # 2) Fetch per-file annotations
        img_name = self.paths[index].stem + ".jpg"
        rows = self._by_file.get(img_name, None)  # DataFrame or None

        # 3) Background case: no rows or all classes NaN
        if rows is None or rows["class"].notna().sum() == 0:
            target = {
                      "image_id": torch.tensor([index], dtype=torch.int64),
                      "labels": torch.empty(0, dtype=torch.int64),
                      "boxes": tv_tensors.BoundingBoxes(torch.empty(0, 4, dtype=torch.float32),
                                                  format="XYXY",
                                                  canvas_size=(H, W),),
                     }
        else:
            # Keep only rows with valid class indices
            rows = rows[rows["class"].notna()]

            # vectorized labels
            labels = torch.as_tensor(rows["class"].to_numpy(), dtype=torch.int64)

            # vectorized boxes
            box_cols = ["xmin", "ymin", "xmax", "ymax"]
            boxes_np = rows[box_cols].to_numpy(dtype="float32", copy=False)
            boxes = torch.from_numpy(boxes_np)  # [N,4], float32

            boxes = tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=(H, W),
            )

            target = {
                "image_id": torch.tensor([index], dtype=torch.int64),
                "labels": labels,
                "boxes": boxes,
            }

        # 4) Apply transforms (they know how to handle tv_tensors)
        if self.transform is not None:
            img, target = self.transform(img, target)

        # 5) Optionally compute areas AFTER transforms, in the transformed space
        if self.area:
            _, H_new, W_new = img.shape
            # target["boxes"] is BoundingBoxes; get underlying tensor view
            boxes_t = target["boxes"].as_subclass(torch.Tensor)
            w = (boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0, max=W_new)
            h = (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0, max=H_new)
            target["areas"] = (w * h).to(torch.float32)

        return img, target


    
    


    def show_with_box(self,
                      index: int,
                      color: str = "g",
                      lw: int = 2,
                      label: bool = False,
                      pred_dict: Dict | None = None,
                      pred_color: str = "r",
                      lw_pred: int = 2,
                      pred_label: bool = False,
                      pred_ref: Literal["size", "normalized", "current"] = "size",
                      pred_size: Tuple[int, int] = (300, 300),  # (H_ref, W_ref) used when pred_ref == "size"
                      ) -> Figure:
        """
        Plots image (determined by index) with or without bounding boxes + labels.
        Predicted bounding boxes + labels can be added as well.

        Inputs
        index: Integer between 0 and the length of the dataset
        color: String denoting the color of ground truth bounding boxes
        lw: Integer for line width of ground truth boxes (lw=0 makes boxes disappear)
        label: Boolean, true will add labels to ground truth bounding boxes
        pred_dict: None or dictionary containing keys 'labels' and 'boxes'
        pred_color: String denoting the color of predicted bounding boxes
        lw_pred: Integer for line width of predicted boxes (lw=0 makes boxes disappear)
        pred_label: Boolean, true will add labels to predicted bounding boxes
        pred_ref:
            - "size": bbox_pred is in pixel coords of a reference frame with size pred_size=(H_ref, W_ref)
            - "normalized": bbox_pred is in [0,1] relative to the displayed image (H,W)
            - "current": bbox_pred already matches the displayed image pixel coords

        Output
        Figure
        """

        # verify index is valid
        if (index > len(self)) | (index < 0):
            raise ValueError(f"Index should be between 0 and {len(self)}, recieved {index}.")
        
        # unpack
        img, target = self[index]

        # ----- convert the image to a numpy array (H,W,C, uint8) -----
        if isinstance(img, Image.Image):
            arr = np.array(img)
        elif isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
        elif isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW -> HWC
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        if arr.dtype.kind == "f" and arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        # ----- figure / axes -----
        H, W = arr.shape[:2]
        dpi = 100
        fig, ax = plt.subplots(1, 1, figsize=(6,6)) # plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax.imshow(arr)  # y downward

        # ----- helper: to numpy float32 (N,4) -----
        def _to_np_xyxy(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().float().numpy()
            else:
                x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = x[None, :]
            assert x.shape[1] == 4, f"Expected (...,4) boxes; got {x.shape}"
            return x

        # ----- draw GT boxes (assumed already in image pixel coords) -----
        gt_boxes = _to_np_xyxy(target["boxes"])
        gt_labels = target.get("labels", None)
        label_dict = self.class_to_idx

        for i in range(gt_boxes.shape[0]):
            x_min, y_min, x_max, y_max = gt_boxes[i]
            # clip
            x_min, y_min = max(0.0, x_min), max(0.0, y_min)
            x_max, y_max = min(W - 1.0, x_max), min(H - 1.0, y_max)
            if not (x_max > x_min and y_max > y_min):
                continue

            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=lw,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            if label and gt_labels is not None:
                # your mapping; convert tensor→int safely
                lab_val = gt_labels[i].item() if isinstance(gt_labels[i], torch.Tensor) else int(gt_labels[i])
                # inverse lookup; raise if not found rather than silently wrong
                try:
                    im_label = next(k for k, v in label_dict.items() if v == lab_val)
                except StopIteration:
                    im_label = str(lab_val)
                ax.text(
                    x_min,
                    y_min,
                    im_label,
                    fontsize=10,
                    color="white",
                    va="bottom",
                    ha="right",
                    bbox=dict(facecolor=color, alpha=0.6, pad=2, edgecolor="none"),
                )

        # ----- predicted boxes: rescale to current image if needed -----
        # if pred_dict is equivalent to (pseudocode) pred_dict is not None and pred_dict is not empty 
        if pred_dict:
            bbox_pred = pred_dict['boxes']
            pb = _to_np_xyxy(bbox_pred)

            if pred_ref == "current":
                # already in the displayed image pixel space
                pb_img = pb
            elif pred_ref == "normalized":
                # [0,1] relative to (W,H)
                sx, sy = float(W), float(H)
                pb_img = pb.copy()
                pb_img[:, [0, 2]] *= sx
                pb_img[:, [1, 3]] *= sy
            elif pred_ref == "size":
                Href, Wref = pred_size
                if Href <= 0 or Wref <= 0:
                    raise ValueError(f"Invalid pred_size={pred_size}. Expect positive (H_ref, W_ref).")
                sx = float(W) / float(Wref)
                sy = float(H) / float(Href)
                pb_img = pb.copy()
                # scale x by sx, y by sy
                pb_img[:, [0, 2]] *= sx
                pb_img[:, [1, 3]] *= sy
            else:
                raise ValueError(f"Unsupported pred_ref={pred_ref}")

            # draw predictions (green)
            for i in range(pb_img.shape[0]):
                x_min, y_min, x_max, y_max = pb_img[i]
                x_min, y_min = max(0.0, x_min), max(0.0, y_min)
                x_max, y_max = min(W - 1.0, x_max), min(H - 1.0, y_max)
                if not (x_max > x_min and y_max > y_min):
                    continue

                rect = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=lw_pred,
                    edgecolor=pred_color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                if pred_label:
                    mask_pred = pred_dict['labels'].tolist()  # 0, 1, ..., C-2
                    # convert from mask to true labels
                    id2name_dict = {v: k for k, v in label_dict.items()}
                    label_pred = []
                    for j in mask_pred:
                        if j in id2name_dict:
                            label_pred.append(id2name_dict[j])
                        else:
                            # raise ValueError(f"Unknown class id {i}")
                            label_pred.append("unknown")
                    if i >= len(label_pred):
                        # mismatch
                        raise IndexError(f"label_pred length {len(label_pred)} < number of boxes {pb_img.shape[0]}")
                    ax.text(
                        x_max,
                        y_max,
                        str(label_pred[i]),
                        fontsize=10,
                        color="white",
                        va="top",
                        ha="left",
                        bbox=dict(facecolor=pred_color, alpha=0.6, pad=2, edgecolor="none"),
                    )

        ax.axis("off")
       # plt.close(fig)
        return fig








def get_file_path_plus_dataframe(targ_dir: str,
                                 rand_seed: int | None = 724,
                                 file_list: list | None = None,
                                 file_pct: float = 1,
                                 ) -> Tuple[list, pd.DataFrame]:
        """
        Creates a list of file paths and corresponding dataframe.

        Inputs:
        targ_dir: Target directory containing images and corresponding dataframe
        rand_seed: (Optional) Random seed that is used when file_pct is not equal to 1
        file_list: (Optional) Custom list of files within targ_dir
        file_pct: Percentage of files to use in targ_dir.  Unused if file_list is given.

        Outputs:
        List of all image file paths
        Dataframe containing classification and bounding box details for each image
        """
        
        if file_list is None:
            all_paths = list(pathlib.Path(targ_dir).glob("*.jpg"))
        else:
            all_paths = [targ_dir / n for n in file_list]
            file_pct = 1
        
        if (file_pct < 0) or (file_pct > 1):
            raise TypeError("file_pct must be between 0 and 1.")
        
        num_files = np.floor((len(all_paths) * file_pct)).astype(int)

        # there should only be one .csv file in the train/test directory, so list(pathlib.Path(targ_dir).glob("*.csv"))[0] gets it!
        # warning just in case
        if len(list(pathlib.Path(targ_dir).glob("*.csv"))) > 1:
            warnings.warn(f"There are multiple .csv files in {targ_dir}, errors with bounding boxes and class labels likely.")
        df = pd.read_csv(list(pathlib.Path(targ_dir).glob("*.csv"))[0])

        if file_pct != 1:
            rng = np.random.default_rng(rand_seed)
            paths = rng.choice(all_paths, size=num_files, replace=False).tolist()
            
            # get file names
            filenames = []
            for file in paths:
                filenames.append(file.stem + '.jpg')
            
            annotate_df = df[df['filename'].isin(filenames)]
        else:
            paths = all_paths
            if file_list is None:
                annotate_df = df
            else:
                annotate_df = df[df['filename'].isin(file_list)]
        
        return paths, annotate_df


def make_train_test_split(full_set: ImageClass,
                          test_size: float = 0.25,
                          rand_state: int | None = 724,
                          transform_train = None,
                          transform_test = None,
                          include_area: bool = False,
                          ) -> Tuple[ImageClass, ImageClass]:
        """
        Create a group stratified train/test split of an image class.

        Inputs:
        full_set - ImageClass file to be train/test splitted
        test_size - float between 0 and 1 that determines the size of the training and testing sets
        rand_state - integer for random state reproducability
        transform_train - torchvision v2 transforms for training set
        transform_test - torchvision v2 transforms for testing set

        Outputs:
        training ImageClass file of size len(full_set) * (1 - test_size)
        testing ImageClass file of size len(full_set) * test_size
        """

        if (test_size <= 0 ) or (test_size >= 1):
            raise ValueError(f"Test size should be a number between 0 and 1, recieved {test_size}.")
        
        # df of original ImageClass file
        df = full_set.annotate_df

        # create stratified group train/test split
        # groups are filenames
        # stratified via class
        groups = df['filename']
        X = df.drop(columns=['class'])
        y = df['class']

        total_splits = np.floor(1 / test_size).astype(int)

        sgkf = StratifiedGroupKFold(n_splits=total_splits, shuffle=True, random_state=rand_state)
        # Take the first fold as train/test split
        tr_idx, te_idx = next(sgkf.split(X, y, groups=groups))

        # create training/testing list of file names
        train_files = df['filename'].iloc[tr_idx].drop_duplicates().to_list()
        test_files = df['filename'].iloc[te_idx].drop_duplicates().to_list()

        # create training/testing ImageClass files
        train_IC = ImageClass(targ_dir=full_set.directory, file_list=train_files, transform=transform_train, include_area=include_area)
        test_IC = ImageClass(targ_dir=full_set.directory, file_list=test_files, transform=transform_test, include_area=include_area)

        return train_IC, test_IC