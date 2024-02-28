from __future__ import annotations

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

    def synthetic_coordinates(self) -> NDArray:
        return self.network.network_coordinates(self.reference_coordinates)

    def _generate_neighbors(self) -> tuple[NDArray, NDArray]:
        """Find the nearest reference neighbors of the synthetic mesh."""
        nn_finder = NearestNeighbors(n_neighbors=self.k)
        nn_finder.fit(self.reference_coordinates)
        return nn_finder.kneighbors(self.synthetic_coordinates(), n_neighbors=self.k)

    def distances(self) -> NDArray:
        return self.neighbors_[0]

    def neighbors(self, *, id_arr: NDArray = None) -> NDArray:
        nn_idx = self.neighbors_[1]
        return id_arr[nn_idx] if id_arr is not None else nn_idx

    def write_synthetic_coordinates(
        self, *, coordinates_fn: str = "synthetic_coordinates.csv"
    ) -> None:
        c = self.synthetic_coordinates()
        cols = [f"CCA{i+1}" for i in range(c.shape[1])]
        idx = [i + 1 for i in range(c.shape[0])]
        df = pd.DataFrame(c, columns=cols, index=idx)
        df.index.name = "SYNTHETIC_PLOT_ID"
        df.to_csv(coordinates_fn, float_format="%.6f")

    def write_neighbors(
        self,
        *,
        neighbors_fn: str = "neighbors.csv",
        distances_fn: str = "distances.csv",
        id_arr: NDArray = None,
    ) -> None:
        distances = self.distances()
        neighbors = self.neighbors(id_arr=id_arr)
        cols = [f"NN{i+1}" for i in range(neighbors.shape[1])]
        idx = [i + 1 for i in range(neighbors.shape[0])]

        def _write_csv(arr: NDArray, fn: str, float_format: str | None = None) -> None:
            df = pd.DataFrame(arr, columns=cols, index=idx)
            df.index.name = "SYNTHETIC_PLOT_ID"
            df.to_csv(fn, float_format=float_format)

        _write_csv(neighbors, neighbors_fn)
        _write_csv(distances, distances_fn, float_format="%.6f")
