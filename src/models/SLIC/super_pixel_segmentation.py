import numpy as np
from skimage.color import rgb2lab
from skimage.util import img_as_float
from scipy.ndimage import sobel
from PIL import Image
from numba import njit


def crop_superpixel(image, labels, segment_id, out_size,
                    pad_value=(124, 116, 104), bbox_pad=0.2):
    img = img_as_float(image).astype(np.float32, copy=False)
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

def _init_centers(lab: np.ndarray, S: int) -> np.ndarray:

    H, W, _ = lab.shape
    S = max(int(S), 1)

    gx = sobel(lab, axis=1)  
    gy = sobel(lab, axis=0)
    gradient = np.sqrt(np.sum(gx * gx + gy * gy, axis=2)).astype(np.float32)

    ny = (H - (S // 2) + S - 1) // S
    nx = (W - (S // 2) + S - 1) // S
    K_est = max(1, int(ny * nx))
    centers = np.empty((K_est, 5), dtype=np.float32)

    idx = 0
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):
            y0 = max(y - 1, 0)
            y1 = min(y + 2, H)
            x0 = max(x - 1, 0)
            x1 = min(x + 2, W)

            region = gradient[y0:y1, x0:x1]
            min_pos = np.argmin(region)
            ry, rx = np.unravel_index(min_pos, region.shape)
            cy = y0 + ry
            cx = x0 + rx

            L, a, b = lab[cy, cx]
            centers[idx, 0] = cy
            centers[idx, 1] = cx
            centers[idx, 2] = L
            centers[idx, 3] = a
            centers[idx, 4] = b
            idx += 1

    return centers[:idx].copy()

@njit(cache=True, fastmath=True)
def _assign_pixels_numba(lab, centers, S, compactness):
    H, W, _ = lab.shape
    K = centers.shape[0]

    labels = -np.ones((H, W), dtype=np.int32)
    dist2 = np.full((H, W), 1e30, dtype=np.float32)

    if S < 1:
        S = 1
    m_over_s2 = (compactness / S) * (compactness / S)

    for k in range(K):
        cy = centers[k, 0]
        cx = centers[k, 1]
        cL = centers[k, 2]
        ca = centers[k, 3]
        cb = centers[k, 4]

        y0 = int(cy - 2 * S)
        y1 = int(cy + 2 * S)
        x0 = int(cx - 2 * S)
        x1 = int(cx + 2 * S)

        if y0 < 0:
            y0 = 0
        if x0 < 0:
            x0 = 0
        if y1 > H:
            y1 = H
        if x1 > W:
            x1 = W

        for y in range(y0, y1):
            dy = y - cy
            for x in range(x0, x1):
                dx = x - cx

                L = lab[y, x, 0]
                a = lab[y, x, 1]
                b = lab[y, x, 2]

                dL = L - cL
                da = a - ca
                db = b - cb
                dc2 = dL * dL + da * da + db * db
                ds2 = dy * dy + dx * dx

                D2 = dc2 + m_over_s2 * ds2

                if D2 < dist2[y, x]:
                    dist2[y, x] = D2
                    labels[y, x] = k

    return labels

@njit(cache=True, fastmath=True)
def _recompute_centers_numba(lab, labels, K, centers_prev):
    H, W, _ = lab.shape
    sums = np.zeros((K, 5), dtype=np.float64)  
    cnt = np.zeros(K, dtype=np.int64)

    for y in range(H):
        for x in range(W):
            k = labels[y, x]
            if k >= 0:
                sums[k, 0] += y
                sums[k, 1] += x
                sums[k, 2] += lab[y, x, 0]
                sums[k, 3] += lab[y, x, 1]
                sums[k, 4] += lab[y, x, 2]
                cnt[k] += 1

    centers = centers_prev.copy()
    for k in range(K):
        if cnt[k] > 0:
            inv = 1.0 / cnt[k]
            centers[k, 0] = sums[k, 0] * inv
            centers[k, 1] = sums[k, 1] * inv
            centers[k, 2] = sums[k, 2] * inv
            centers[k, 3] = sums[k, 3] * inv
            centers[k, 4] = sums[k, 4] * inv

    return centers


@njit(cache=True)
def _enforce_connectivity_numba(labels, min_size):
    H, W = labels.shape
    new_labels = -np.ones((H, W), dtype=np.int32)
    visited = np.zeros((H, W), dtype=np.uint8)

    stack_y = np.empty(H * W, dtype=np.int32)
    stack_x = np.empty(H * W, dtype=np.int32)
    comp_y = np.empty(H * W, dtype=np.int32)
    comp_x = np.empty(H * W, dtype=np.int32)

    max_adj = 512
    adj_labels = np.empty(max_adj, dtype=np.int32)
    adj_counts = np.empty(max_adj, dtype=np.int32)

    current = 0

    for y0 in range(H):
        for x0 in range(W):
            if visited[y0, x0] != 0:
                continue

            old_lbl = labels[y0, x0]
            visited[y0, x0] = 1

            sp = 0
            stack_y[sp] = y0
            stack_x[sp] = x0
            sp += 1

            comp_n = 0
            adj_n = 0 

            while sp > 0:
                sp -= 1
                y = stack_y[sp]
                x = stack_x[sp]

                comp_y[comp_n] = y
                comp_x[comp_n] = x
                comp_n += 1
                ny = y - 1
                nx = x
                if ny >= 0:
                    if visited[ny, nx] == 0 and labels[ny, nx] == old_lbl:
                        visited[ny, nx] = 1
                        stack_y[sp] = ny
                        stack_x[sp] = nx
                        sp += 1
                    elif labels[ny, nx] != old_lbl:
                        nl = new_labels[ny, nx]
                        if nl >= 0:
                            found = False
                            for i in range(adj_n):
                                if adj_labels[i] == nl:
                                    adj_counts[i] += 1
                                    found = True
                                    break
                            if (not found) and adj_n < max_adj:
                                adj_labels[adj_n] = nl
                                adj_counts[adj_n] = 1
                                adj_n += 1

                ny = y + 1
                nx = x
                if ny < H:
                    if visited[ny, nx] == 0 and labels[ny, nx] == old_lbl:
                        visited[ny, nx] = 1
                        stack_y[sp] = ny
                        stack_x[sp] = nx
                        sp += 1
                    elif labels[ny, nx] != old_lbl:
                        nl = new_labels[ny, nx]
                        if nl >= 0:
                            found = False
                            for i in range(adj_n):
                                if adj_labels[i] == nl:
                                    adj_counts[i] += 1
                                    found = True
                                    break
                            if (not found) and adj_n < max_adj:
                                adj_labels[adj_n] = nl
                                adj_counts[adj_n] = 1
                                adj_n += 1

                ny = y
                nx = x - 1
                if nx >= 0:
                    if visited[ny, nx] == 0 and labels[ny, nx] == old_lbl:
                        visited[ny, nx] = 1
                        stack_y[sp] = ny
                        stack_x[sp] = nx
                        sp += 1
                    elif labels[ny, nx] != old_lbl:
                        nl = new_labels[ny, nx]
                        if nl >= 0:
                            found = False
                            for i in range(adj_n):
                                if adj_labels[i] == nl:
                                    adj_counts[i] += 1
                                    found = True
                                    break
                            if (not found) and adj_n < max_adj:
                                adj_labels[adj_n] = nl
                                adj_counts[adj_n] = 1
                                adj_n += 1
                ny = y
                nx = x + 1
                if nx < W:
                    if visited[ny, nx] == 0 and labels[ny, nx] == old_lbl:
                        visited[ny, nx] = 1
                        stack_y[sp] = ny
                        stack_x[sp] = nx
                        sp += 1
                    elif labels[ny, nx] != old_lbl:
                        nl = new_labels[ny, nx]
                        if nl >= 0:
                            found = False
                            for i in range(adj_n):
                                if adj_labels[i] == nl:
                                    adj_counts[i] += 1
                                    found = True
                                    break
                            if (not found) and adj_n < max_adj:
                                adj_labels[adj_n] = nl
                                adj_counts[adj_n] = 1
                                adj_n += 1
            if comp_n < min_size and adj_n > 0:
        
                best_i = 0
                best_c = adj_counts[0]
                for i in range(1, adj_n):
                    if adj_counts[i] > best_c:
                        best_c = adj_counts[i]
                        best_i = i
                target = adj_labels[best_i]
            else:
                target = current
                current += 1

            for i in range(comp_n):
                new_labels[comp_y[i], comp_x[i]] = target

    return new_labels


def slic_segmentation(image, n_segments=100, compactness=20, max_iter=10,
                      min_size_factor=0.5, early_stop_eps=1e-3):

    image = img_as_float(image).astype(np.float32, copy=False)
    H, W, _ = image.shape
    N = H * W

    lab = rgb2lab(image).astype(np.float32, copy=False)

    S = int(np.sqrt(N / max(n_segments, 1)))
    S = max(S, 1)

    centers = _init_centers(lab, S).astype(np.float32, copy=False)
    K = centers.shape[0]

    comp = np.float32(compactness)

    for _ in range(max_iter):
        labels = _assign_pixels_numba(lab, centers, S, comp)
        new_centers = _recompute_centers_numba(lab, labels, K, centers)

        if early_stop_eps > 0.0:
            shift = float(np.max(np.abs(new_centers - centers)))
            centers = new_centers
            if shift < early_stop_eps:
                break
        else:
            centers = new_centers

    labels = _assign_pixels_numba(lab, centers, S, comp)

    min_size = int(min_size_factor * S * S)
    if min_size < 1:
        min_size = 1
    labels = _enforce_connectivity_numba(labels, min_size)

    return labels
