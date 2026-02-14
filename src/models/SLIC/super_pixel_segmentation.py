import numpy as np
from skimage.color import rgb2lab
from skimage.util import img_as_float
from scipy.ndimage import sobel
from PIL import Image

def crop_superpixel(image, labels, segment_id, out_size,
                        pad_value=(124, 116, 104), bbox_pad=0.2):
    img = img_as_float(image)
    H, W = labels.shape

    mask = (labels == segment_id)
    if not np.any(mask):
        return None

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    if bbox_pad > 0:
        bh = y1 - y0
        bw = x1 - x0
        py = int(round(bh * bbox_pad))
        px = int(round(bw * bbox_pad))
        y0 = max(0, y0 - py)
        y1 = min(H, y1 + py)
        x0 = max(0, x0 - px)
        x1 = min(W, x1 + px)

    crop = img[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]

    if isinstance(pad_value, (tuple, list, np.ndarray)):
        pad_rgb_255 = np.array(pad_value, dtype=np.float32)
    else:
        pad_rgb_255 = np.array([pad_value, pad_value, pad_value], dtype=np.float32)

    pad_rgb_01 = pad_rgb_255 / 255.0 

    crop[~crop_mask] = pad_rgb_01

    ch, cw = crop.shape[:2]
    scale = min(out_size / max(cw, 1), out_size / max(ch, 1))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    crop_uint8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
    pil_crop = Image.fromarray(crop_uint8, mode="RGB").resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (out_size, out_size),
                       tuple(int(round(x)) for x in pad_rgb_255))

    paste_x = (out_size - new_w) // 2
    paste_y = (out_size - new_h) // 2
    canvas.paste(pil_crop, (paste_x, paste_y))

    patch = np.asarray(canvas).astype(np.float32) / 255.0  

    return patch


def _init_centers(lab, S):
    H, W, _ = lab.shape
    gradient = np.zeros((H, W), dtype=np.float64)
    
    for i in range(3):
        gx = sobel(lab[:, :, i], axis=1)
        gy = sobel(lab[:, :, i], axis=0)
        gradient += gx * gx + gy * gy
    gradient = np.sqrt(gradient)

    centers = []
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):
            y0 = max(y - 1, 0)
            y1 = min(y + 2, H)
            x0 = max(x - 1, 0)
            x1 = min(x + 2, W)

            region = gradient[y0:y1, x0:x1]
            min_idx = np.unravel_index(np.argmin(region), region.shape)
            cy = y0 + min_idx[0]
            cx = x0 + min_idx[1]

            L, a, b = lab[cy, cx]
            centers.append([float(cy), float(cx), float(L), float(a), float(b)])

    return np.array(centers, dtype=np.float64)


def _enforce_connectivity(labels, min_size=20):
    H, W = labels.shape
    new_labels = -np.ones((H, W), dtype=np.int32)
    visited = np.zeros((H, W), dtype=bool)

    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    current = 0

    for y0 in range(H):
        for x0 in range(W):
            if visited[y0, x0]:
                continue

            old_lbl = labels[y0, x0]
            stack = [(y0, x0)]
            visited[y0, x0] = True

            component = []
            adjacent_new = []

            while stack:
                y, x = stack.pop()
                component.append((y, x))

                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if (not visited[ny, nx]) and labels[ny, nx] == old_lbl:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                        elif labels[ny, nx] != old_lbl:
                            nl = new_labels[ny, nx]
                            if nl >= 0:
                                adjacent_new.append(nl)

            comp_size = len(component)

            if comp_size < min_size and adjacent_new:
                adjacent_new = np.asarray(adjacent_new, dtype=np.int32)
                target = np.bincount(adjacent_new).argmax()
            else:
                target = current
                current += 1

            for (y, x) in component:
                new_labels[y, x] = target

    _, inv = np.unique(new_labels, return_inverse=True)
    return inv.reshape(H, W).astype(np.int32)


def _assign_pixels(lab, centers, S, compactness):
    H, W, _ = lab.shape
    K = len(centers)

    labels = -np.ones((H, W), dtype=np.int32)
    dist2 = np.full((H, W), np.inf, dtype=np.float64)

    m_over_s2 = (compactness / max(S, 1)) ** 2

    for k, (cy, cx, cL, ca, cb) in enumerate(centers):
        y0 = max(int(cy - 2 * S), 0)
        y1 = min(int(cy + 2 * S), H)
        x0 = max(int(cx - 2 * S), 0)
        x1 = min(int(cx + 2 * S), W)

        for y in range(y0, y1):
            for x in range(x0, x1):
                L, a, b = lab[y, x]
                dc2 = (L - cL) ** 2 + (a - ca) ** 2 + (b - cb) ** 2
                ds2 = (y - cy) ** 2 + (x - cx) ** 2
                D2 = dc2 + m_over_s2 * ds2

                if D2 < dist2[y, x]:
                    dist2[y, x] = D2
                    labels[y, x] = k

    return labels


def slic_segmentation(image, n_segments=100, compactness=20, max_iter=10, min_size_factor=0.5):
    image = img_as_float(image)
    H, W, _ = image.shape
    N = H * W

    lab = rgb2lab(image)

    S = int(np.sqrt(N / max(n_segments, 1)))
    S = max(S, 1)

    centers = _init_centers(lab, S)
    K = len(centers)

    ys, xs = np.indices((H, W))

    for _ in range(max_iter):
        labels = _assign_pixels(lab, centers, S, compactness)

        new_centers = centers.copy()  
        for k in range(K):
            mask = (labels == k)
            if np.any(mask):
                new_centers[k, 0] = ys[mask].mean()
                new_centers[k, 1] = xs[mask].mean()
                new_centers[k, 2:] = lab[mask].mean(axis=0)

        centers = new_centers

    labels = _assign_pixels(lab, centers, S, compactness)

    min_size = int(min_size_factor * S * S)
    labels = _enforce_connectivity(labels, min_size=min_size)

    return labels
