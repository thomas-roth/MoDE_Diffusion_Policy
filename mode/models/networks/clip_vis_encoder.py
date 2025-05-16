from typing import List

import torch
import torch.nn as nn
from PIL import Image
from mode.models.networks.clip import load_clip


class VisClip(nn.Module):
    def __init__(self, freeze_backbone: bool = True, model_name: str = "ViT-B/16"):
        super(VisClip, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = f"CLIP_{model_name}"

        # Load CLIP model
        print(f"loading vision CLIP model with backbone: {model_name}")
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_vit.parameters():
                param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        self.clip_vit, self.clip_vit_preprocess = load_clip(model_name, device=self.device)
        self.output_dim = self.clip_vit.visual.output_dim

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        images = [Image.fromarray(image.to(torch.uint8).cpu().numpy()).convert("RGB") for image in images]
        with torch.no_grad():
            preprocessed_images = []
            for image in images:
                preprocessed_image = self.clip_vit_preprocess(image).to(self.device)
                preprocessed_images.append(preprocessed_image)
            preprocessed_images = torch.stack(preprocessed_images)
            embedded_images = self.clip_vit.encode_image(preprocessed_images)
        return torch.unsqueeze(embedded_images, 1)
