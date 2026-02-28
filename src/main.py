import numpy as np
import torch
import os
from pathlib import Path
from PIL import Image
from models.CNN.model import load_model
from models.CNN.extract_embedding import extract_superpixel_embedding
from models.SLIC.super_pixel_segmentation import slic_segmentation, crop_superpixel
from sklearn.cluster import KMeans # будет заменен на свой
import matplotlib.pyplot as plt
from tqdm import tqdm


TARGET_CLASS = 0       
NUM_IMAGES = 50            
RESOLUTIONS = [15, 50, 80] 
NUM_CONCEPTS_K = 25
COMPACTNESS = 20
BACKGROUND_FILTER = 0.005

model, device = load_model()
data_dir = Path(f"src/datasets/golden_set/{TARGET_CLASS}")
images_of_class = []
for img_path in data_dir.glob("*"):
    pil_img = Image.open(img_path).convert("RGB")
    img_np = np.array(pil_img)        
    images_of_class.append((img_np, TARGET_CLASS))
    if len(images_of_class) >= NUM_IMAGES:
        break

all_embeddings = []
all_patches = [] 

# eval уже прописан в load model
with torch.no_grad():
    for img_np, label in tqdm(images_of_class):
        for n_seg in RESOLUTIONS:
            labels = slic_segmentation(img_np, n_segments=n_seg, compactness=COMPACTNESS)
            unique_labels = np.unique(labels)
            for seg_id in unique_labels:
                patch = crop_superpixel(img_np, labels, segment_id=seg_id, out_size=128)
                if patch is None:
                    continue
                if np.var(patch) < BACKGROUND_FILTER: 
                    continue
                
                emb = extract_superpixel_embedding(patch, model=model, device=device)
                emb_np = emb.cpu().detach().numpy().flatten()
                
                all_embeddings.append(emb_np)
                all_patches.append(patch)
                
X = np.array(all_embeddings)
print(f"\nИзвлечено полезных патчей: {X.shape[0]}")

norms = np.linalg.norm(X, axis=1, keepdims=True)
norms[norms == 0] = 1
X_normalized = X / norms

kmeans = KMeans(n_clusters=NUM_CONCEPTS_K, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(X_normalized)
centroids = kmeans.cluster_centers_

print("Отрисовка результатов...")

output_dir = Path("saved_concepts")
output_dir.mkdir(exist_ok=True)

num_examples = 6

for i in range(NUM_CONCEPTS_K):
    idx_in_cluster = np.where(cluster_labels == i)[0]
    
    if len(idx_in_cluster) == 0:
        continue

    cluster_features = X_normalized[idx_in_cluster]
    distances = np.linalg.norm(cluster_features - centroids[i], axis=1)
    best_idx_sorted = np.argsort(distances)
    
    top_indices = idx_in_cluster[best_idx_sorted[:num_examples]]
    actual_num_examples = len(top_indices)
    
    fig, axes = plt.subplots(1, actual_num_examples, figsize=(2.5 * actual_num_examples, 2.5))
    if actual_num_examples == 1:
        axes = [axes]
        
    for j, patch_idx in enumerate(top_indices):
        ax = axes[j]
        ax.imshow(all_patches[patch_idx])
        ax.axis("off")
        
        if j == 0:
            count = len(idx_in_cluster)
            ax.set_title(f"Concept {i}\n(Total: {count} patches)", fontsize=12, loc='left', pad=10)

    plt.tight_layout()

    save_path = output_dir / f"concept_{i:02d}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    
print(f"\nВсе концепты сохранены в папку: {output_dir.absolute()}")