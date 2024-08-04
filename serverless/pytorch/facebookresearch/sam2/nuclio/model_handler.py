# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "./sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, self.sam_checkpoint, device="cuda"))

    def handle(self, image):
        self.predictor.set_image(np.array(image))
        features = self.predictor.get_image_embedding()
        return features