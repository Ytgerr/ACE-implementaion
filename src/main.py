import numpy as np
import torch
import torchvision
from PIL import Image

from models.CNN.model import load_model

model = load_model()

print(model)
