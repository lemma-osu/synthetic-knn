from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from synthetic_knn.point_networks import PointNetwork


class SyntheticPlots:
    def __init__(
        self, reference_coordinates: NDArray, network: PointNetwork, k: int = 10
    ):
        self.reference_coordinates = reference_coordinates
        self.network = network
        self.k = k
        self.neighbors_ = self._generate_neighbors()

    @staticmethod
    def _create_column_names(n: int, prefix: str) -> list[str]:
        return [f"{prefix}{i+1}" for i in range(n)]

    def _create_dataframe(self, arr: NDArray, prefix: str = "NN") -> pd.DataFrame:
        df = pd.DataFrame(
            arr,
            columns=self._create_column_names(arr.shape[1], prefix),
            index=np.arange(1, arr.shape[0] + 1),
        )
        df.index.name = "SYNTHETIC_PLOT_ID"
        return df

    def synthetic_coordinates(
        self, as_frame: bool = False, prefix: str = "AXIS"
    ) -> NDArray:
        coordinates = self.network.network_coordinates(self.reference_coordinates)
        if as_frame:
            return self._create_dataframe(coordinates, prefix=prefix)
        return coordinates

    def _generate_neighbors(self) -> tuple[NDArray, NDArray]:
        """Find the nearest reference neighbors of the synthetic mesh."""
        nn_finder = NearestNeighbors(n_neighbors=self.k)
        nn_finder.fit(self.reference_coordinates)
        return nn_finder.kneighbors(self.synthetic_coordinates(), n_neighbors=self.k)

    def distances(self, as_frame: bool = False) -> NDArray:
        distances = self.neighbors_[0]
        if as_frame:
            return self._create_dataframe(distances)
        return distances

    def neighbors(self, *, id_arr: NDArray = None, as_frame: bool = False) -> NDArray:
        nn_idx = self.neighbors_[1]
        neighbors = id_arr[nn_idx] if id_arr is not None else nn_idx
        if as_frame:
            return self._create_dataframe(neighbors)
        return neighbors
