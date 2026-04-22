from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from PIL import Image
import io
import base64

import numpy as np
from skimage.segmentation import mark_boundaries
from models.SLIC.super_pixel_segmentation import slic_segmentation
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from mainshow import (
    BACKGROUND_FILTER,
    COMPACTNESS,
    N_CAVS,
    RESOLUTIONS,
    evaluate_concepts_clusters,
    evaluate_embeddings_with_patches,
    extract_acts_and_grads,
    extract_images_by_classes,
    form_concepts,
)
from models.CNN.model import load_model

from torchvision.transforms import v2
import torch

from models.TCAV.main import Concept, extract_cav, prepare_concept_dataset

model, device = None, None

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(size=(128, 128)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device
    model, device = load_model()
    print("Model Fully Loaded!")

    yield

    print("Application shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/slic-processing")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # Get segments and boundaries
    segments = slic_segmentation(image, n_segments=50)
    segmented_image = mark_boundaries(image, segments)  # [0,1] float64

    # Convert to uint8 and create PIL Image
    segmented_image_uint8 = (segmented_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(segmented_image_uint8)

    # Save to buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format="jpeg")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="image/jpeg")


@app.get("/extract_clusters_for_class")
async def extract_clusters(
    class_index: str, n_images: int, n_concepts: int, n_cavs: int, n_examples: int
):
    global device, model

    data_dir = Path(f"datasets/golden_set/{class_index}")

    images_of_class = extract_images_by_classes(data_dir, class_index, n_images)

    all_embeddings, all_patches, all_img_ids = evaluate_embeddings_with_patches(
        model, images_of_class, RESOLUTIONS, COMPACTNESS, BACKGROUND_FILTER, device
    )

    (
        cluster_ids,
        patches,
        image_ids,
        embeddings,
        cluster_labels,
    ) = evaluate_concepts_clusters(all_embeddings, all_patches, all_img_ids)

    patches_embeddings, _ = extract_acts_and_grads(
        model, transforms(patches), 0, device
    )

    concepts: list[Concept] = form_concepts(
        cluster_ids, patches, cluster_labels, patches_embeddings, image_ids
    )

    resulting_concepts = []
    tcav_scores = []

    for concept_idx, concept in enumerate(concepts[:n_concepts]):
        resulting_concepts.append([])

        render_examples = concept.images[:n_examples]

        for idx, patch in enumerate(render_examples):
            img_patch = Image.fromarray((patch * 255).astype(np.uint8))

            buffer = io.BytesIO()
            img_patch.save(buffer, format="JPEG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            data_url = f"data:image/jpeg;base64,{img_base64}"

            resulting_concepts[-1].append(data_url)

    _, grads = extract_acts_and_grads(
        model,
        transforms([image_class[0] for image_class in images_of_class]),
        0,
        device,
    )

    grads = np.array(grads)

    for idx in range(len(concepts)):
        main_concept, other_concepts = prepare_concept_dataset(concepts, idx)

        cavs = np.array(
            [
                extract_cav(main_concept, other_concepts, seed=int(seed))
                for seed in np.random.randint(0, 10000, n_cavs)
            ]
        )

        results = []

        for grad in grads:
            scores = cavs @ grad
            results.extend(scores > 0)

        tcav_scores.append(round(np.mean(results), 2))

    return [resulting_concepts, tcav_scores]
