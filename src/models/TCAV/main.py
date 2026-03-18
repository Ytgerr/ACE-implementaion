from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Concept:
    id: int
    images: np.ndarray
    name: Optional[str]

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return len(self.images)


def construct_concepts() -> list[Concept]:
    """
    Converts raw concepts format from k-means into classes
    """
    ...


def take_representation(model, layer, concept_inst) -> np.ndarray:
    """
    Takes vector representation from linear layer of a model
    """
    ...


def extract_cav(concept, other_concepts):
    """
    Extracts concept activation vector for some concept
    """
    ...


def extract_cav_noisy(concept):
    """
    Extracts concept activation vector with respect to random noise vectors, not real ones
    """
    ...


def directional_derivative(image, cav):
    """
    Computes score for how cav affects on prediction of image
    """
    ...


def tcav():
    """
    Tests how concept affects on prediction
    """
    ...
