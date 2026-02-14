import numpy as np
import torch
import torchvision
from PIL import Image
from models.CNN.model import load_model
from models.CNN.extract_embedding import extract_superpixel_embedding
from models.SLIC.super_pixel_segmentation import slic_segmentation

model, device = load_model()
ds_raw = torchvision.datasets.Imagenette("../../datasets", split="val", download=True, transform=None)
pil_img, target = ds_raw[0]         
image = np.array(pil_img) 

labels = slic_segmentation(image, n_segments=50, compactness=20)

seg_id = int(np.random.choice(np.unique(labels)))

emb = extract_superpixel_embedding()

print("class target:", target)
print("segment id:", seg_id)
print("embedding:", emb)