import math
from typing import Dict, List

import torch
import torchvision
import torch.nn as nn

from myPatchEmb import myPatchEmbedding
from myPosEnc import myPositionEncoding, PositionEncoding2D
from myTransEncoder import myTransformerEncoderLayer
from myTransDecoder import myTransformerDecoderLayer
from TransEncDec import *


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    """
    def __init__(self,
                 class_to_idx_dict: Dict,
                 img_size: int = 300,
                 patch_H: int = 20,
                 patch_W: int = 15,
                 in_channels: int = 3,
                 embed_dim: int = 128,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 num_queries: int = 100):
        """
        Inputs:
         - img_size: Size of input image (assumed square).
         - patch_H: Height of each patch.
         - patch_W: Width of each patch.
         - in_channels: Number of image channels.
         - embed_dim: Embedding dimension for each patch.
         - num_layers: Number of Transformer encoder layers.
         - num_heads: Number of attention heads.
         - dim_feedforward: Hidden size of feedforward network.
         - dropout: Dropout probability.
         - num_queries: Number of learned object queries.
        """
        super().__init__()
        self.num_classes = len(class_to_idx_dict)
        self.num_queries = num_queries
        self.img_size = img_size

        self.class_to_idx = class_to_idx_dict
        self.idx_to_class = {v: k for k, v in class_to_idx_dict.items()}

        self.patch_embed = myPatchEmbedding(img_size=img_size,
                                            patch_H=patch_H,
                                            patch_W=patch_W,
                                            in_channels=in_channels,
                                            embed_dim=embed_dim)

        self.positional_encoding = PositionEncoding2D(embed_dim=embed_dim,
                                                      grid_height=img_size // patch_H,
                                                      grid_width=img_size // patch_W,
                                                      dropout=dropout)

        encoder_layer = myTransformerEncoderLayer(input_dim=embed_dim,
                                                  num_heads=num_heads,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout)
        
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=num_layers,
                                          embed_dim=embed_dim)

        self.query_embed = nn.Embedding(num_queries, embed_dim)

        decoder_layer = myTransformerDecoderLayer(input_dim=embed_dim,
                                                  num_heads=num_heads,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout)

        self.decoder = TransformerDecoder(decoder_layer=decoder_layer,
                                          num_layers=num_layers,
                                          embed_dim=embed_dim)

        # Final classification layer to predict class scores from final decoder layer.
        self.class_head = nn.Linear(embed_dim, self.num_classes+1)

        # Final localization layer to predict bounding boxes from final decoder layer.
        self.loc_head = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim, embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim, 4)
                                      )

        self.apply(self._init_weights)

        # Treat the localization head output as an offset from evenly spaced
        # query-box priors.  This buffer is derived from num_queries, so it does
        # not need to be saved and older checkpoints remain compatible.
        self.register_buffer(
            "query_box_logits",
            self._make_query_box_logits(num_queries),
            persistent=False,
        )

        # Start with zero offsets so an untrained model predicts the priors
        # exactly.  The final layer begins learning offsets on the first step.
        nn.init.zeros_(self.loc_head[-1].weight)
        nn.init.zeros_(self.loc_head[-1].bias)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _make_query_box_logits(num_queries: int) -> torch.Tensor:
        """Create near-square grid priors in normalized ``cxcywh`` logits."""
        num_cols = math.ceil(math.sqrt(num_queries))
        num_rows = math.ceil(num_queries / num_cols)

        centers = []
        queries_per_row, rows_with_extra_query = divmod(num_queries, num_rows)
        for row in range(num_rows):
            queries_in_row = queries_per_row + (row < rows_with_extra_query)
            y = (row + 0.5) / num_rows
            for col in range(queries_in_row):
                x = (col + 0.5) / queries_in_row
                centers.append((x, y))

        centers = torch.tensor(centers, dtype=torch.float32)
        sizes = torch.full_like(centers, 0.5)
        return torch.cat((centers, sizes), dim=-1).logit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Vision Transformer.

        Inputs:
         - x: Input image tensor of shape (B, C, H, W)

        Returns:
         - class_logits: Output classification logits of shape (L, B, Q, K+1).
         - bboxes: Output bounding box locations of shape (L, B, Q, 4) in (c_x, c_y, w, h) format.
         L is the number of decoder layers.
        """
        # ------------------------------------------------------------
        # Image content
        # ------------------------------------------------------------

        # (B, C, H, W) -> (B, N, D)
        patches = self.patch_embed(x)

        B, N, D = patches.shape

        # ------------------------------------------------------------
        # Image positional encoding
        # ------------------------------------------------------------

        # self.positional_encoding.pe has shape (1, N, D)
        pos = self.positional_encoding.pe.to(
            device=patches.device,
            dtype=patches.dtype,
        )

        # (1, N, D) -> (B, N, D)
        pos = pos.expand(B, -1, -1)

        # ------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------

        memory = self.encoder(src=patches,
                              pos=pos,)

        # ------------------------------------------------------------
        # Decoder queries
        # ------------------------------------------------------------

        # Learned query positional embeddings:
        # (Q, D) -> (B, Q, D)
        query_pos = self.query_embed.weight.unsqueeze(0)
        query_pos = query_pos.expand(B, -1, -1)

        # Decoder query CONTENT initially contains no information.
        tgt = torch.zeros_like(query_pos)

        # ------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------

        # decoder_outputs: (L, B, Q, D)
        decoder_output = self.decoder(tgt=tgt,
                                      memory=memory,
                                      query_pos=query_pos,
                                      memory_pos=pos,)

        # ------------------------------------------------------------
        # Heads
        # ------------------------------------------------------------

        # (L, B, Q, K+1)
        class_logits = self.class_head(decoder_output)

        # (L, B, Q, 4)
        bbox_offsets = self.loc_head(decoder_output)
        query_box_logits = self.query_box_logits.to(dtype=bbox_offsets.dtype)
        bboxes = torch.sigmoid(bbox_offsets + query_box_logits)

        return class_logits, bboxes



    @torch.inference_mode()
    def predict(self,
                images: torch.Tensor,
                pre_class_logits: torch.Tensor | None = None,
                pre_bboxes: torch.Tensor | None = None,
                conf_thresh: float | None = None,
                ) -> List[Dict]:
        """
        Inputs:
        - images: Tensor of shape [B, 3, H, W].
        - pre_class_logits: Optional tensor of shape [B, Q, K+1],
            containing precomputed class logits.
        - pre_bboxes: Optional tensor of shape [B, Q, 4],
            containing precomputed normalized bounding boxes in
            (cx, cy, w, h) format.
        - conf_thresh: Optional float to set confidence threshold cutoff.

        Returns:
        - predictions: List of length B. Each element is a dictionary:
            {
                "labels": Tensor of shape [Q], containing values 0, 1, ..., K-1,
                "scores": Tensor of shape [Q], containing class probabilities,
                "boxes":  Tensor of shape [Q, 4], containing bounding boxes
                        in pixel-space (x1, y1, x2, y2) format.
            }
        """

        B, _, H, W = images.shape

        # ------------------------------------------------------------
        # Obtain model predictions
        # ------------------------------------------------------------

        if pre_class_logits is None or pre_bboxes is None:

            was_training = self.training
            self.eval()

            class_logits, pred_bboxes = self(images)

            # class_logits, pred_bboxes contain all decoder layers,
            # so [-1] gets the final layer.
            class_logits = class_logits[-1]
            pred_bboxes = pred_bboxes[-1]

            if was_training:
                self.train()

        else:
            class_logits = pre_class_logits
            pred_bboxes = pre_bboxes

        # ------------------------------------------------------------
        # Convert logits -> probabilities
        #
        # Shape:
        #   (B, Q, K+1) -> (B, Q, K+1)
        # ------------------------------------------------------------

        class_probs = torch.softmax(class_logits, dim=-1)

        # For every query, find the most probable class.
        #
        # scores: (B, Q)
        # labels: (B, Q)
        scores, labels = torch.max(class_probs[..., :self.num_classes], dim=-1)

        # ------------------------------------------------------------
        # Convert bounding boxes:
        #
        # normalized (cx, cy, w, h)
        #       ->
        # normalized (x1, y1, x2, y2)
        # ------------------------------------------------------------

        boxes_xyxy = torchvision.ops.box_convert(
            pred_bboxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        # ------------------------------------------------------------
        # Convert normalized coordinates to pixel coordinates
        # ------------------------------------------------------------

        scale = torch.tensor(
            [W, H, W, H],
            dtype=boxes_xyxy.dtype,
            device=boxes_xyxy.device,
        )

        boxes_xyxy = boxes_xyxy * scale

        # Clip boxes to image boundaries.
        boxes_xyxy[..., 0::2] = boxes_xyxy[..., 0::2].clamp(0, W)
        boxes_xyxy[..., 1::2] = boxes_xyxy[..., 1::2].clamp(0, H)

        # ------------------------------------------------------------
        # Construct output for each image
        # ------------------------------------------------------------

        predictions = []

        # Probability of background/no-object
        if conf_thresh is not None:
            background_scores = class_probs[..., self.num_classes]

        for i in range(B):
            keep = None
            # Optionally impose an additional confidence threshold
            # Query must prefer a foreground class over background
            if conf_thresh is not None:
                keep = (scores[i] >= conf_thresh) & (scores[i] > background_scores[i])

            predictions.append(
                {
                    "labels": labels[i] if keep is None else labels[i][keep],
                    "scores": scores[i] if keep is None else scores[i][keep],
                    "boxes": boxes_xyxy[i] if keep is None else boxes_xyxy[i][keep],
                }
            )

        return predictions

        
