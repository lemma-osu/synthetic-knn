from __future__ import annotations

from enum import IntEnum

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


class ClassifierEnum(IntEnum):
    QUANTILE = 1
    EQUAL_INTERVAL = 2


def quantile_bins(arr: np.ndarray, n_bins: int = 10) -> list[np.ndarray]:
    """Return quantile bins for each column of an array"""
    q = np.linspace(0.0, 1.0, n_bins + 1)
    return [np.quantile(arr[:, i], q) for i in range(arr.shape[1])]


def equal_interval_bins(arr: np.ndarray, n_bins: int = 10) -> list[np.ndarray]:
    """Return equal interval bins for each column of an array"""
    return [
        np.linspace(min(arr[:, i]), max(arr[:, i]), n_bins + 1)
        for i in range(arr.shape[1])
    ]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MeshCoords:
    arr: np.ndarray
    n_bins: int
    classifier: ClassifierEnum = ClassifierEnum.QUANTILE

    def to_coords(self) -> np.ndarray:
        """Return the mesh midpoint coordinates as a n-D array"""
        func_dict = {
            ClassifierEnum.EQUAL_INTERVAL: equal_interval_bins,
            ClassifierEnum.QUANTILE: quantile_bins,
        }
        bins = func_dict[self.classifier](self.arr, self.n_bins)
        coords = np.array([(b[:-1] + b[1:]) / 2.0 for b in bins])
        return np.vstack(list(map(np.ravel, np.meshgrid(*coords, indexing="xy")))).T
