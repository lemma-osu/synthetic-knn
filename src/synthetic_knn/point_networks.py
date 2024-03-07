"""
Module to generate point networks for synthetic kNN data.  Point networks are
used to generate synthetic data for kNN imputation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class PointNetwork(ABC):
    """Abstract base class for point networks."""

    @abstractmethod
    def network_coordinates(self, reference_coordinates: Sequence | NDArray) -> NDArray:
        """Return the point network coordinates as a n-D array."""
        ...


class ReferenceNetwork(PointNetwork):
    """Point network that uses the reference coordinates as the network coordinates."""

    def network_coordinates(self, reference_coordinates: Sequence | NDArray) -> NDArray:
        return reference_coordinates


class FuzzedNetwork(PointNetwork):
    """Point network that 'fuzzes' the reference coordinates by a random distance
    between the minimum_distance and the maximum_distance."""

    def __init__(
        self,
        minimum_distance: float = 0.1,
        maximum_distance: float = 0.5,
    ):
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance

    def _fuzz_points(self, coordinates: NDArray) -> NDArray:
        """Generate fuzzed points in n-D space by adding a random offset
        to the original points."""
        # Generate a random direction vector
        rng = np.random.default_rng()
        direction = rng.standard_normal(coordinates.shape)

        # Normalize the direction vector
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)

        # Scale the direction vector by a random distance between
        # min_distance and max_distance
        distance = rng.uniform(
            self.minimum_distance, self.maximum_distance, (len(coordinates), 1)
        )
        offset = direction * distance

        # Add the offset to the original points
        return coordinates + offset

    def network_coordinates(self, reference_coordinates: Sequence | NDArray) -> NDArray:
        return self._fuzz_points(np.asarray(reference_coordinates))


class Mesh(PointNetwork):
    """Base class for mesh point networks."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    @abstractmethod
    def generate_bin_edges(self, reference_coordinates: NDArray) -> NDArray:
        """Return the edges of the bins."""
        ...

    @staticmethod
    def generate_bin_midpoints_mesh(bins: NDArray) -> NDArray:
        """Find midpoints of bins in each dimension to serve as mesh points."""
        midpoint_coordinates = (bins[:, :-1] + bins[:, 1:]) / 2.0
        return np.column_stack(
            list(map(np.ravel, np.meshgrid(*midpoint_coordinates, indexing="xy")))
        )

    def network_coordinates(self, reference_coordinates: Sequence | NDArray) -> NDArray:
        reference_coordinates = np.asarray(reference_coordinates)
        if self.n_bins ** reference_coordinates.shape[1] > 1e6:
            raise ValueError(
                "The number of mesh points is set to "
                f"{self.n_bins} ** {reference_coordinates.shape[1]} "
                "and exceeds 1e6. Please reduce the number of bins or "
                "the number of features (components) in the reference "
                "coordinates."
            )
        bin_edges = self.generate_bin_edges(np.asarray(reference_coordinates))
        return self.generate_bin_midpoints_mesh(bin_edges)


class QuantileMesh(Mesh):
    """Mesh point network with quantile bin spacing in each dimension."""

    def generate_bin_edges(self, reference_coordinates: NDArray) -> NDArray:
        q = np.linspace(0.0, 1.0, self.n_bins + 1)
        return np.quantile(reference_coordinates, q, axis=0).T


class EqualIntervalMesh(Mesh):
    """Mesh point network with equal-interval bin spacing in each dimension."""

    def generate_bin_edges(self, reference_coordinates: NDArray) -> NDArray:
        min_vals = np.min(reference_coordinates, axis=0)
        max_vals = np.max(reference_coordinates, axis=0)
        return np.linspace(min_vals, max_vals, num=self.n_bins + 1).T
