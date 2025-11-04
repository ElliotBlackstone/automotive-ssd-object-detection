import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
import pathlib
from PIL import Image
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split

# Write a custom dataset class (inherits from torch.utils.data.Dataset)


# 1. Subclass torch.utils.data.Dataset
class ImageClass(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self,
                 targ_dir: str,
                 file_list: list = None,
                 transform=None,
                 file_pct: float = 1,
                 rand_seed: int = 724,
                 ) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        if file_list is None:
            all_paths = list(pathlib.Path(targ_dir).glob("*.jpg"))
        else:
            all_paths = [targ_dir / n for n in file_list]
            file_pct = 1
        
        if (file_pct < 0) | (file_pct > 1):
            raise TypeError("file_pct must be between 0 and 1.")
        
        num_files = np.floor((len(all_paths) * file_pct)).astype(int)

        # there should only be one .csv file in the train/test directory, so list(pathlib.Path(targ_dir).glob("*.csv"))[0] gets it!
        df = pd.read_csv(list(pathlib.Path(targ_dir).glob("*.csv"))[0])

        if file_pct != 1:
            rng = np.random.default_rng(rand_seed)
            self.paths = rng.choice(all_paths, size=num_files, replace=False).tolist()
            
            # get file names
            filenames = []
            for file in self.paths:
                filenames.append(file.stem + '.jpg')
            
            self.annotate_df = df[df['filename'].isin(filenames)]
        else:
            self.paths = all_paths
            if file_list is None:
                self.annotate_df = df
            else:
                self.annotate_df = df[df['filename'].isin(file_list)]
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        # self.classes, self.class_to_idx = find_classes(targ_dir)
        # there should only be one .csv file in the train/test directory, so list(pathlib.Path(targ_dir).glob("*.csv"))[0] gets it!
        # self.annotate_df = pd.read_csv(list(pathlib.Path(targ_dir).glob("*.csv"))[0])
        self.classes = list(self.annotate_df['class'].unique())
        self.classes.sort() # alphabetical sort of classes

        # move background/empty to front of list
        try:
            index = self.classes.index('empty')
        except ValueError:
            print(f"'{'empty'}' not found in the list.")
        else:
            removed_element = self.classes.pop(index)
            self.classes.insert(0, removed_element)


        self.class_to_idx = dict(zip(self.classes, range(0, len(self.classes))))
        self.directory = targ_dir

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns one sample of data, in the form img, target, where
        img is a torch.Tensor and target is a Dictonary of the form
        target = {
                  "boxes":   Tensor[n_i, 4]  # float32, xyxy, absolute pixels
                  "labels":  Tensor[n_i]     # int64, in {1..num_classes}
                  "image_id": Tensor[1]      # int64 unique id i.e. index
                  "area":    Tensor[n_i]     # float32 (box area in pixels)
                  "iscrowd": Tensor[n_i]     # int64 (0 or 1), 0 if you do not use crowd
                }
        where n_i is the number of objects in the ith image.
        """
        img = decode_image(str(self.paths[index]))
        # img = self.load_image(index).convert("RGB")
        # image_path = self.paths[index]
        img_name = self.paths[index].stem + '.jpg'
        img_df = self.annotate_df[self.annotate_df['filename'] == img_name]
        img_df.loc[:, 'class'] = img_df['class'].map(self.class_to_idx)
        
        # obj_dict = img_df[['class', 'xmin', 'xmax', 'ymin', 'ymax']].to_dict(orient='records')
        # class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        # class_idx = self.class_to_idx[class_name]

        # initialize boxes, labels, areas
        boxes = torch.zeros(len(img_df), 4)
        labels = torch.zeros(len(img_df), dtype=torch.int64)
        areas = torch.zeros(len(img_df))
        # populate via loop
        for i in range(len(img_df)):
            j = 0
            labels[i] = torch.tensor(img_df.iloc[i]['class'], dtype=torch.int64)

            for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
                boxes[i, j] = img_df.iloc[i][coord]
                j = j + 1

            # areas[i] = (boxes[i,2] - boxes[i,0]).clamp(min=0) * (boxes[i,3] - boxes[i,1]).clamp(min=0)

        iscrowd = torch.zeros(len(img_df), dtype=torch.int64)

        # Wrap into tv_tensors so transforms know how to move boxes with the image
        _, H, W = img.shape
        boxes = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=(H, W)
        )

        target = {
                  'image_id': torch.tensor([index], dtype=torch.int64),
                  'labels': labels,
                  'boxes': boxes,
                  'areas': areas,
                  'iscrowd': iscrowd,
                  }
        
        if self.transform is not None:
            img, target = self.transform(img, target)

        target['areas'] = (target['boxes'][:,2] - target['boxes'][:,0]).clamp(0, W) * (target['boxes'][:,3] - target['boxes'][:,1]).clamp(0, H)

        # if the image is background
        if img_df['class'].isin([0]).sum() != 0:
            target = {
                  'image_id': torch.tensor([index], dtype=torch.int64),
                  'labels': torch.zeros((0,), dtype=torch.int64),
                  'boxes': torch.zeros((0, 4), dtype=torch.float32),
                  'areas': torch.zeros((0,), dtype=torch.float32),
                  'iscrowd': torch.zeros((0,), dtype=torch.int64),
                  }

        # Convert back to plain tensors for model consumption
        # target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        # img = torch.as_tensor(img, dtype=torch.uint8)  # or later cast to float/normalize

        return img, target # {**{"file_name": img_name}, **{"target": obj_dict}}
    

    
    def show_with_box(self,
                      index: int,
                      color: str = "C0",
                      lw: int = 2,
                      label: bool = False,
                      pred_box: bool = False,
                      bbox_pred: torch.Tensor = torch.zeros((1,4))
                      ) -> Figure:
        
        # convert the image to a numpy array
        img, target = self[index]

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
        

        # plot image
        H, W = arr.shape[:2]
        dpi = 100
        fig, ax = plt.subplots(figsize=(W/dpi, H/dpi), dpi=dpi)
        ax.imshow(arr)  # origin='upper' -> y downward, matches image coords

        for i in range(len(target['labels'])):

            # basic sanity + clipping
            x_min = target['boxes'][i, 0]
            y_min = target['boxes'][i, 1]
            x_max = target['boxes'][i, 2]
            y_max = target['boxes'][i, 3]
            x_min, y_min = max(0.0, x_min), max(0.0, y_min)
            x_max, y_max = min(W - 1, x_max), min(H - 1, y_max)
            if not (x_max > x_min and y_max > y_min):
                # raise ValueError("Degenerate or inverted box after clipping.")
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
            if label:
                im_label_masked = target['labels'][i]
                im_label = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(im_label_masked)]
                ax.text(
                    x_min, y_min,
                    str(im_label),
                    fontsize=10,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(facecolor=color, alpha=0.6, pad=2, edgecolor="none"),
                )
        
        if pred_box == True:
            for i in range(len(bbox_pred)):
                x_min = bbox_pred[i, 0]
                y_min = bbox_pred[i, 1]
                x_max = bbox_pred[i, 2]
                y_max = bbox_pred[i, 3]
                x_min, y_min = max(0.0, x_min), max(0.0, y_min)
                x_max, y_max = min(W - 1, x_max), min(H - 1, y_max)
                if not (x_max > x_min and y_max > y_min):
                    # raise ValueError("Degenerate or inverted box after clipping.")
                    continue

                
                rect = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=lw,
                    edgecolor='g',
                    facecolor="none",
                )
                ax.add_patch(rect)
        
        ax.axis("off")

        plt.close(fig)
        return fig


def make_train_test_split(full_set: ImageClass,
                          test_size: float = 0.25,
                          rand_state: int = None,
                          transform_train = None,
                          transform_test = None,
                          ) -> Tuple[ImageClass, ImageClass]:
        """
        Create a train/test split of an image class.

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
        # df of original ImageClass file
        df = full_set.annotate_df

        # sklearn train/test split the original dataset, stratify with respect to class
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=rand_state, stratify=df['class'])

        # create training/testing list of file names
        train_files = train_df['filename'].to_list()
        test_files = test_df['filename'].to_list()

        # create training/testing ImageClass files
        train_IC = ImageClass(targ_dir=full_set.directory, file_list=train_files, transform=transform_train)
        test_IC = ImageClass(targ_dir=full_set.directory, file_list=test_files, transform=transform_test)

        return train_IC, test_IC




# 1. Subclass torch.utils.data.Dataset
class ImageClassSimple(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        # self.classes, self.class_to_idx = find_classes(targ_dir)
        # there should only be one .csv file in the train/test directory, so list(pathlib.Path(targ_dir).glob("*.csv"))[0] gets it!
        self.annotate_df = pd.read_csv(list(pathlib.Path(targ_dir).glob("*.csv"))[0])
        self.classes = ['car', 'empty']
        self.class_to_idx = {'car': 0, 'empty': 11}

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns one sample of data, in the form img, dict, where
        img is a torch.Tensor and dict is a Dictonary of the form
        {
          "file_name": "img_0001.jpg",
          "objects": [
             {'class': '1', 'xmin': 186, 'xmax': 192, 'ymin': 251, 'ymax': 258},
             {'class': '4', 'xmin': 80, 'xmax': 85, 'ymin': 250, 'ymax': 267},
             ...
           ]
        }.
        """
        img = self.load_image(index)
        # image_path = self.paths[index]
        img_name = self.paths[index].stem + '.jpg'
        img_df = self.annotate_df[self.annotate_df['filename'] == img_name]
        img_df.loc[:, 'class'] = img_df['class'].map(self.class_to_idx)
        
        obj_dict = img_df[['class', 'xmin', 'xmax', 'ymin', 'ymax']].dropna(subset=['class']).to_dict(orient='records')

        # Transform if necessary
        if self.transform:
            # is Resize part of the transform?
            resize_flag = False
            resize_index = 0
            for i, x in enumerate(self.transform.transforms):
                if x.__class__.__name__ == "Resize":
                    resize_flag = True
                    resize_index = i

            if resize_flag == False:
                return self.transform(img), {**{"file_name": img_name}, **{"objects": obj_dict}}
            else:
                for column in ['xmin', 'xmax']:
                    new_width = self.transform.transforms[resize_index].size[0]
                    img_df.loc[:, column] = img_df[column] * (new_width / img_df['width'])
                for column in ['ymin', 'ymax']:
                    new_height = self.transform.transforms[resize_index].size[1]
                    img_df.loc[:, column] = img_df[column] * (new_height / img_df['height'])
                
                obj_dict = img_df[['class', 'xmin', 'xmax', 'ymin', 'ymax']].to_dict(orient='records')

                return self.transform(img), {**{"file_name": img_name}, **{"objects": obj_dict}}

        else:
            return img, {**{"file_name": img_name}, **{"objects": obj_dict}}
    



    def show_with_box(self, index: int, color: str = "C0", lw: int = 2, label: bool = False) -> Tuple[Figure, Axes]:
        

        # convert the image to a numpy array
        img = self.load_image(index)

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
        


        # plot image
        fig, ax = plt.subplots()
        ax.imshow(arr)  # origin='upper' -> y downward, matches image coords

        for i in range(len(self[index][1]['objects'])):

            # basic sanity + clipping
            H, W = arr.shape[:2]
            x_min = self[index][1]['objects'][i]['xmin']
            x_max = self[index][1]['objects'][i]['xmax']
            y_min = self[index][1]['objects'][i]['ymin']
            y_max = self[index][1]['objects'][i]['ymax']
            x_min, y_min = max(0.0, x_min), max(0.0, y_min)
            x_max, y_max = min(W - 1, x_max), min(H - 1, y_max)
            if not (x_max > x_min and y_max > y_min):
                raise ValueError("Degenerate or inverted box after clipping.")

            
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=lw,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            if label:
                im_label_masked = self[index][1]['objects'][i]['class']
                im_label = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(im_label_masked)]
                ax.text(
                    x_min, y_min,
                    str(im_label),
                    fontsize=10,
                    color="white",
                    va="bottom",
                    ha="right",
                    bbox=dict(facecolor=color, alpha=0.6, pad=2, edgecolor="none"),
                )
            ax.axis("off")

        return fig, ax