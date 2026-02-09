import numpy as np
from skimage.color import rgb2lab
from skimage.util import img_as_float
from scipy.ndimage import sobel


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
