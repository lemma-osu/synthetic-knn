from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Union

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


class ClassifierEnum(IntEnum):
    QUANTILE = 1
    EQUAL_INTERVAL = 2


def quantile_bins(arr: NDArray, n_bins: int = 10) -> list[NDArray]:
    """Return quantile bins for each column of an array"""
    q = np.linspace(0.0, 1.0, n_bins + 1)
    return [np.quantile(arr[:, i], q) for i in range(arr.shape[1])]


def equal_interval_bins(arr: NDArray, n_bins: int = 10) -> list[NDArray]:
    """Return equal interval bins for each column of an array"""
    return [
        np.linspace(min(arr[:, i]), max(arr[:, i]), n_bins + 1)
        for i in range(arr.shape[1])
    ]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MeshCoords:
    reference_coordinates: Union[Sequence, NDArray]  # noqa: UP007
    n_bins: int = 5
    classifier: ClassifierEnum = ClassifierEnum.QUANTILE

    def to_coords(self) -> NDArray:
        """Return the mesh midpoint coordinates as a n-D array"""
        if self.classifier == ClassifierEnum.QUANTILE:
            bins = quantile_bins(self.reference_coordinates, self.n_bins)
        else:
            bins = equal_interval_bins(self.reference_coordinates, self.n_bins)
        midpoint_coordinates = np.array([(b[:-1] + b[1:]) / 2.0 for b in bins])
        return np.vstack(
            list(map(np.ravel, np.meshgrid(*midpoint_coordinates, indexing="xy")))
        ).T
