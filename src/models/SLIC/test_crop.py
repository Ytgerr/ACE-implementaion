from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from src.models.SLIC.super_pixel_segmentation import slic_segmentation, crop_superpixel

def show_random_slic_segment(image, n_segments=50, compactness=20, max_iter=10,
                             min_size_factor=0.5, out_size=128, pad_value=(124, 116, 104), bbox_pad=0.1, seed=None):

 
    rng = np.random.default_rng(seed)

    labels = slic_segmentation(image, n_segments=n_segments, compactness=compactness,
                               max_iter=max_iter, min_size_factor=min_size_factor)

    seg_ids = np.unique(labels)
    seg_id = int(rng.choice(seg_ids))

    patch = crop_superpixel(
        image, labels,
        segment_id=seg_id,
        out_size=out_size,
        pad_value=pad_value,
        bbox_pad=bbox_pad,
    )

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image)
    ax[0].set_title("Original image")
    ax[0].axis("off")

    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]

    overlay = image.copy()
    if overlay.dtype != np.uint8:
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    overlay[boundaries] = [255, 0, 0]

    ax[1].imshow(overlay)
    ax[1].set_title(f"SLIC boundaries (random seg_id={seg_id})")
    ax[1].axis("off")

    ax[2].imshow(patch)
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

    return None


image = imread("src/models/SLIC/test_image/acula.jpg")  
show_random_slic_segment(image, seed=60)