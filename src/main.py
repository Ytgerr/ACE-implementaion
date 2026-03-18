from typing import Literal, Tuple

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from models.CNN.model import load_model
from models.CNN.extract_embedding import extract_superpixel_embedding
from models.SLIC.super_pixel_segmentation import slic_segmentation, crop_superpixel
from models.KMeans import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

TARGET_CLASS = 0  # class from golden set(available 0, 1, 2)
NUM_IMAGES = 50  # number of images
RESOLUTIONS = [15, 50, 80]  # Size of super pixel
NUM_CONCEPTS_K = 25  # number of clusters
TOP_SEGMENTS = 40  # keep top N segments per cluster

# SLIC parameter
# Higher — more regular / pixels square-like (grid-like).
# Lower — more irregular / pixels boundary-adaptive (contour-following).
COMPACTNESS = 20
BACKGROUND_FILTER = 0.009  # threshhold for background superpixels

MIN_IMAGE_RATIO = 0.1  # minimum fraction of images a cluster should appear in (10%)
MIN_CLUSTER_SIZE = 10  # minimum segments in a cluster

RENDER_EXAMPLES = 6

type ClassImage = Tuple[np.ndarray, int]


def extract_images_by_classes(
    images_dir: Path, target_class: int, n_inst: int = 50
) -> list[ClassImage]:
    """
    Extracts images from `images_dir` by selected `target_class` up to `n_inst` instances
    """
    images_of_class = []

    for img_path in images_dir.glob("*"):
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        images_of_class.append((img_np, target_class))

        if len(images_of_class) >= n_inst:
            break

    return images_of_class


def evaluate_embeddings_with_patches(
    model,
    cls_images: list[ClassImage],
    resolutions: list[int],
    compactness: int,
    bg_filter: float,
    device: Literal["cpu", "cuda"] = "cpu",
) -> Tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Extracts patches from `cls_images` using SLIC segmentation algorithm, then embeddes them using `model`

    Returns:
    - Embeddings for extracted patches
    - Extracted patches themselfs
    - Images ids which patch coomes from

    """
    embeddings = []
    patches = []
    images_ids = []

    with torch.no_grad():
        for img_id, (img_np, _) in enumerate(tqdm(cls_images)):
            for n_seg in resolutions:
                labels = slic_segmentation(
                    img_np, n_segments=n_seg, compactness=compactness
                )
                unique_labels = np.unique(labels)
                for seg_id in unique_labels:
                    patch = crop_superpixel(
                        img_np, labels, segment_id=seg_id, out_size=128
                    )
                    if patch is None:
                        continue
                    if np.var(patch) < bg_filter:
                        continue

                    emb = extract_superpixel_embedding(
                        patch, model=model, device=device
                    )
                    emb_np = emb.cpu().detach().numpy().flatten()

                    embeddings.append(emb_np)
                    patches.append(patch)
                    images_ids.append(img_id)

    return embeddings, patches, images_ids


def evaluate_concepts_clusters(
    embeddings: list[np.ndarray],
    patches: list[np.ndarray],
    ids: list[int],
    n_clusters: int = 25,
    n_images: int = 50,
    top_filter: int = 40,
    min_ratio: float = 0.1,
    min_cluster_size: int = 10,
) -> tuple[list[int], list[np.ndarray], list[int], list[np.ndarray], list[int]]:
    """
    Evaluates clusters for concepts and filters them

    Returns:
    - Passed clusters IDs
    - Filtered patches
    - Filtered patches sources (images ids)
    - Filtered patches embeddings
    - Filtered patches cluster labels
    """

    X = np.array(embeddings)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_normalized = X / norms

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    cluster_labels = kmeans.fit_predict(emb_normalized)
    centroids = kmeans.cluster_centers_

    if (cluster_labels is None) or (centroids is None):
        raise RuntimeError("Can't define clusters")

    filtered_indices = []

    # Sorting elements by distance from centroids and cut off top N
    for i in range(n_clusters):
        idx_in_cluster = np.where(cluster_labels == i)[0]
        if len(idx_in_cluster) == 0:
            continue

        cluster_features = emb_normalized[idx_in_cluster]
        distances = np.linalg.norm(cluster_features - centroids[i], axis=1)
        best_idx_sorted = np.argsort(distances)
        keep = idx_in_cluster[best_idx_sorted[:top_filter]]
        filtered_indices.extend(keep)

    filtered_embeddings = emb_normalized[filtered_indices]
    filtered_cluster_labels = cluster_labels[filtered_indices]

    # Taking top N patches for centroids with their IDs
    filtered_patches = [patches[i] for i in filtered_indices]
    filtered_images_ids = [ids[i] for i in filtered_indices]

    # Filter clusters indeces by presence in images and their size
    min_images = int(n_images * min_ratio)
    valid_clusters = []
    for i in range(n_clusters):
        idx = np.where(filtered_cluster_labels == i)[0]
        if len(idx) == 0:
            continue
        unique_imgs = len(set(filtered_images_ids[j] for j in idx))
        if unique_imgs >= min_images:
            valid_clusters.append(i)

    final_clusters = []

    for i in valid_clusters:
        idx = np.where(filtered_cluster_labels == i)[0]
        if len(idx) >= min_cluster_size:
            final_clusters.append(i)

    return (
        final_clusters,
        filtered_patches,
        filtered_images_ids,
        filtered_embeddings,
        filtered_cluster_labels,
    )


def form_concepts(clusters_ids, patches, cluster_labels):
    concepts = []

    for cluster_id in clusters_ids:
        concept_indices = np.where(cluster_labels == cluster_id)[0]
        concepts.append([patches[idx] for idx in concept_indices])

    return concepts


def render_concepts(concepts, render_num):
    for concept_idx, concept in enumerate(concepts):
        render_examples = concept[:render_num]
        actual_num_examples = len(render_examples)

        fig, axes = plt.subplots(
            1, actual_num_examples, figsize=(2.5 * actual_num_examples, 2.5)
        )

        if actual_num_examples == 1:
            axes = [axes]

        for idx, patch in enumerate(render_examples):
            ax = axes[idx]
            ax.imshow(patch)
            ax.axis("off")

            if idx == 0:
                count = len(concept)
                ax.set_title(
                    f"Concept {concept_idx}\n(Total: {count} patches)",
                    fontsize=12,
                    loc="left",
                    pad=10,
                )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model, device = load_model()

    data_dir = Path(f"src/datasets/golden_set/{TARGET_CLASS}")

    images_of_class = extract_images_by_classes(data_dir, TARGET_CLASS, 10)

    all_embeddings, all_patches, all_img_ids = evaluate_embeddings_with_patches(
        model, images_of_class, RESOLUTIONS, COMPACTNESS, BACKGROUND_FILTER, "cuda"
    )

    (
        cluster_ids,
        patches,
        ids,
        embeddings,
        cluster_labels,
    ) = evaluate_concepts_clusters(all_embeddings, all_patches, all_img_ids)

    concepts = form_concepts(cluster_ids, patches, cluster_labels)

    render_concepts(concepts, 10)
