import torch
import torchvision.transforms.v2 as v2

from PIL import Image, ImageDraw, ImageFont, ImageOps
from .SSD_from_scratch import mySSD


def show_prediction_side_by_side(model: mySSD,
                                 image_path: str | None,
                                 pil_img: Image.Image | None,
                                 score_thresh: float = 0.2,
                                 nms_thresh: float = 0.5,
                                 max_per_img: int = 100,
                                 class_agnostic: bool = False,
                                 target_height: int = 512,
                                 ) -> Image.Image:
        """
        Load an image from disk, run model.predict on it, and return a new image with
        two panels side by side:
        - left:  original image (resized to height = target_height, width chosen to
                preserve the original aspect ratio)
        - right: same image with predicted bounding boxes + labels + scores

        Parameters
        ----------
        model : mySSD
            The mySSD model to run the prediction
        image_path : str | None
            Path to the input image file. Mutually exclusive with `pil_img`.
        pil_img : PIL.Image.Image | None
            Pre-loaded PIL image. Mutually exclusive with `image_path`.
        score_thresh : float, optional (default=0.2)
            Score threshold passed to model.predict.
        nms_thresh : float, optional (default=0.5)
            NMS IoU threshold passed to model.predict.
        max_per_img : int, optional (default=100)
            Maximum number of detections per image (passed to model.predict).
        class_agnostic : bool, optional (default=False)
            Whether to perform class-agnostic NMS in model.predict.
        target_height : int, optional (default=512)
            Desired display height. The display width is chosen to preserve the
            original aspect ratio after EXIF correction.

        Returns
        -------
        combined_image : PIL.Image.Image
            A PIL image of size (target_height, 2 * out_w). The left half is the
            original resized image, the right half is the annotated image.
        """

        if ((image_path is not None) and (pil_img is not None)) or \
        ((image_path is None) and (pil_img is None)):
            raise TypeError(
                "An image path or PIL image should be supplied, not both or neither. "
                f"Received image path {image_path} and "
                f"PIL image {None if pil_img is None else 'img received'}."
            )

        device = next(model.parameters()).device
        class_to_idx = model.class_to_idx
        idx_to_class = model.idx_to_class

        # -------------------------------------------------------------------------
        # 1. Load original image (and fix orientation)
        # -------------------------------------------------------------------------
        if image_path is not None:
            pil_orig = Image.open(image_path).convert("RGB")
        else:
            pil_orig = pil_img

        pil_orig = ImageOps.exif_transpose(pil_orig)

        # Get original (width, height) after EXIF correction
        orig_w, orig_h = pil_orig.size

        # -------------------------------------------------------------------------
        # 2. Model input preprocessing (fixed 300x300 for the SSD)
        # -------------------------------------------------------------------------
        model_size = (300, 300)  # (width, height) for PIL

        preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((300, 300), antialias=True),  # (height, width)
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = preprocess(pil_orig)                  # [3, 300, 300]
        img_tensor = img_tensor.unsqueeze(0).to(device)    # [1, 3, 300, 300]

        # -------------------------------------------------------------------------
        # 3. Run prediction
        # -------------------------------------------------------------------------
        preds = model.predict(images=img_tensor,
                              score_thresh=score_thresh,
                              nms_thresh=nms_thresh,
                              max_per_img=max_per_img,
                              class_agnostic=class_agnostic,
                              pre_loc_all=None,
                              pre_conf_all=None)

        pred = preds[0]
        boxes = pred["boxes"].to("cpu")    # [K,4], xyxy in 300x300 coords
        labels = pred["labels"].to("cpu")  # [K]
        scores = pred["scores"].to("cpu")  # [K]

        # -------------------------------------------------------------------------
        # 4. Choose display size with fixed height and aspect-preserving width
        # -------------------------------------------------------------------------
        # Fix output height
        out_h = target_height

        # Preserve aspect ratio: out_w / out_h = orig_w / orig_h
        if orig_h == 0:
            raise ValueError("Original image has zero height; cannot compute aspect ratio.")
        aspect = orig_w / orig_h
        out_w = max(1, int(round(out_h * aspect)))  # ensure at least 1 pixel wide

        # Resize original for display
        # IMPORTANT: PIL expects (width, height)
        pil_disp = pil_orig.resize((out_w, out_h), Image.LANCZOS)

        # -------------------------------------------------------------------------
        # 5. Annotate a copy of the display image
        # -------------------------------------------------------------------------
        line_width = 2
        font_size = 14

        annotated = pil_disp.copy()
        draw = ImageDraw.Draw(annotated)

        model_w, model_h = model_size  # (300, 300)

        # Scale boxes from 300x300 model space to (out_w, out_h) display space
        scale_x = out_w / model_w
        scale_y = out_h / model_h

        boxes_disp = boxes.clone()
        boxes_disp[:, [0, 2]] *= scale_x
        boxes_disp[:, [1, 3]] *= scale_y

        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except OSError:
            font = ImageFont.load_default()

        for box, label, score in zip(boxes_disp, labels, scores):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

            cls_idx = int(label)
            cls_str = idx_to_class.get(cls_idx, str(cls_idx))
            text = f"{cls_str}"

            # bbox of text when baseline is at (0, 0)
            text_box = draw.textbbox((0, 0), text, font=font)
            tw = text_box[2] - text_box[0]
            th = text_box[3] - text_box[1]   # total text height
            ymin = text_box[1]               # usually negative

            text_x = x1
            text_top = max(y1 - th, 0)       # desired *top* of text background

            # Baseline y so that the text's top is at text_top
            baseline_y = text_top - ymin

            # Background rectangle exactly covering the text bbox
            draw.rectangle(
                [text_x, text_top, text_x + tw, text_top + th],
                fill="red"
            )
            # Draw text with correct baseline
            draw.text((text_x, baseline_y), text, fill="white", font=font)

        # -------------------------------------------------------------------------
        # 6. Concatenate left (original) and right (annotated) panels
        # -------------------------------------------------------------------------
        left_panel = pil_disp
        right_panel = annotated

        combined = Image.new("RGB", (2 * out_w, out_h))
        combined.paste(left_panel, (0, 0))
        combined.paste(right_panel, (out_w, 0))

        return combined