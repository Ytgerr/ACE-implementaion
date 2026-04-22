import time
import numpy as np
from skimage import io, segmentation
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from src.models.SLIC.super_pixel_segmentation import slic_segmentation

image = io.imread("src/models/SLIC/test_image/ptica.jpg")


def bench(fn, n=1, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return out, np.array(times)


fn_sk = lambda: segmentation.slic(image, n_segments=50, start_label=0)
fn_my = lambda: slic_segmentation(image, n_segments=50)

segments_sk, t_sk = bench(fn_sk, n=5, warmup=1)
segments_my, t_my = bench(fn_my, n=5, warmup=1)

print(
    f"skimage slic: mean={t_sk.mean():.4f}s  std={t_sk.std():.4f}s  min={t_sk.min():.4f}s"
)
print(
    f"my slic    : mean={t_my.mean():.4f}s  std={t_my.std():.4f}s  min={t_my.min():.4f}s"
)

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
