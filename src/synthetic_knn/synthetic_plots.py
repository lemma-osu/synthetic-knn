from __future__ import annotations

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors

from .mesh import ClassifierEnum, MeshCoords


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SyntheticPlots:
    coordinates: np.ndarray
    n_components: int | None = None
    n_bins: int = 10
    classifier: ClassifierEnum = ClassifierEnum.QUANTILE
    k: int = 7

    def __post_init__(self):
        if self.n_components is None:
            n_coord_components = self.coordinates.shape[1]
            self.n_components = (
                n_coord_components
                if self.n_components is None
                else min(n_coord_components, self.n_components)
            )

    def _generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Find the nearest reference neighbors of the synthetic mesh"""
        nn_finder = NearestNeighbors(n_neighbors=self.k)
        nn_finder.fit(self.reference_coordinates)
        return nn_finder.kneighbors(self.synthetic_coordinates, n_neighbors=self.k)

    @property
    def reference_coordinates(self) -> np.ndarray:
        return self.coordinates[:, : self.n_components]

    @property
    def synthetic_coordinates(self) -> np.ndarray:
        return MeshCoords(
            arr=self.reference_coordinates,
            n_bins=self.n_bins,
            classifier=self.classifier,
        ).to_coords()

    @property
    def distances(self) -> np.ndarray:
        return self._generate()[0]

    def neighbors(self, *, id_arr: np.ndarray = None) -> np.ndarray:
        nn_idx = self._generate()[1]
        return id_arr[nn_idx] if id_arr is not None else nn_idx
