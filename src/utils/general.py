import math
from numbers import Number
from typing import Sequence, Tuple

import numpy.typing as npt
from sklearn.model_selection import train_test_split


def set_seed(random_seed: Number) -> None:
    from avalanche.training.determinism.rng_manager import RNGManager

    RNGManager.set_random_seeds(random_seed)


def split_evenly_by_classes(
    X: Sequence[Tuple[npt.ArrayLike, Number, Number]],
    y: Sequence[Number],
    train_ratio: float,
    shuffle: bool = True,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=1 - train_ratio,
        stratify=y,
        shuffle=shuffle,
    )

    return X_train, X_test, y_train, y_test


def label_transform(x: float | int, n_labels: int = 10) -> int:
    # we know that the range spans between 0.0 and 100.0
    # we can divide them equally right away based on `n_labels`

    if x <= 0.0:  # edge cases
        return 0

    if x >= 100.0:
        return n_labels - 1

    divider = 100.0 / n_labels

    return min(math.ceil(x / divider) - 1, n_labels - 1)


def custom_round(x, base: float = 10):
    return int(base * round(float(x) / base))


def ceil_up(x, base: float = 10):
    return math.ceil(x / base) * base


__all__ = [
    "set_seed",
    "split_evenly_by_classes",
    "label_transform",
    "custom_round",
    "ceil_up",
]
