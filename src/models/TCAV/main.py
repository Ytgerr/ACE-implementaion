import numpy as np
from dataclasses import dataclass
from typing import Optional
import random as rn
from sklearn.linear_model import LogisticRegression


@dataclass
class Concept:
    id: int
    images: list[np.ndarray]
    representations: list[np.ndarray]
    source_ids: list[int]
    name: Optional[str] = None

    def __getitem__(self, idx):
        return (self.images[idx], self.representations[idx])

    def __len__(self):
        return len(self.images)

    def get_random_repr(self):
        return rn.choice(self.representations)


def prepare_concept_dataset(concepts: list[Concept], idx: int, seed: int = 42):
    """
    Prepares balanced dataset for concept under defined index
    """
    rn.seed(seed)

    all_concepts = concepts.copy()
    needed_concept = all_concepts.pop(idx).representations

    other_concepts = []

    while len(other_concepts) < len(needed_concept):
        other_concepts.append(rn.choice(all_concepts).get_random_repr())

    return needed_concept, other_concepts


def extract_cav(
    main_concept: list[np.ndarray], other_concepts: list[np.ndarray], seed: int = 42
):
    """
    Extracts concept activation vector for some concept
    """
    rn.seed(seed)

    X = np.vstack([main_concept, other_concepts])

    y = np.array([1] * len(main_concept) + [0] * len(other_concepts))

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    cav = clf.coef_[0]
    cav = cav / np.linalg.norm(cav)

    return cav


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
