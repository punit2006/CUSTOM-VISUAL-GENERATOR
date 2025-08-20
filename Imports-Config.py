# -*- coding: utf-8 -*-
"""Config and imports for Medical Image Generator"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# Common negative prompts for medical images
NEGATIVE_PROMPT = (
    "colorful, cartoon, artistic, painting, illustration, fantasy, anime, "
    "bright colors, vibrant, artistic style, creative, decorative"
)
