from skimage import io, segmentation, color
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from src.models.SLIC.super_pixel_segmentation import slic_segmentation

image = io.imread('src/models/SLIC/test_image/ptica.jpg')
segments_sk = segmentation.slic(image, n_segments=50)
segments_my = slic_segmentation(image, n_segments=50)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("skimage SLIC")
plt.imshow(mark_boundaries(image, segments_sk))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("My SLIC")
plt.imshow(mark_boundaries(image, segments_my))
plt.axis("off")

plt.show()