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

    def fit(self, reference_coordinates: Sequence | NDArray):
        """Set the reference coordinates for the network."""
        self.reference_coordinates = np.array(reference_coordinates)
        return self

    @abstractmethod
    def network_coordinates(self) -> NDArray:
        """Return the point network coordinates as a n-D array."""
        ...


class ReferenceNetwork(PointNetwork):
    """Point network that uses the reference coordinates as the network coordinates."""

    def network_coordinates(self) -> NDArray:
        return self.reference_coordinates


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

    def _fuzz_points(self) -> NDArray:
        """Generate fuzzed points in n-D space by adding a random offset
        to the original points."""
        coords = self.reference_coordinates

        # Generate a random direction vector
        rng = np.random.default_rng()
        direction = rng.standard_normal(coords.shape)

        # Normalize the direction vector
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)

        # Scale the direction vector by a random distance between
        # min_distance and max_distance
        distance = rng.uniform(
            self.minimum_distance, self.maximum_distance, (len(coords), 1)
        )
        offset = direction * distance

        # Add the offset to the original points
        return coords + offset

    def network_coordinates(self) -> NDArray:
        """Generate a network of fuzzed points in n-D space."""
        return self._fuzz_points()


class Mesh(PointNetwork):
    """Base class for mesh point networks."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    @abstractmethod
    def generate_bin_edges(self):
        """Return the edges of the bins."""
        ...

    @staticmethod
    def generate_bin_midpoints_mesh(bins: NDArray) -> NDArray:
        """Find midpoints of bins in each dimension to serve as mesh points."""
        midpoint_coordinates = (bins[:, :-1] + bins[:, 1:]) / 2.0
        return np.column_stack(
            list(map(np.ravel, np.meshgrid(*midpoint_coordinates, indexing="xy")))
        )

    def network_coordinates(self) -> NDArray:
        bin_edges = self.generate_bin_edges()
        return self.generate_bin_midpoints_mesh(bin_edges)


class QuantileMesh(Mesh):
    """Mesh point network with quantile bin spacing in each dimension."""

    def generate_bin_edges(self) -> NDArray:
        q = np.linspace(0.0, 1.0, self.n_bins + 1)
        return np.quantile(self.reference_coordinates, q, axis=0).T


class EqualIntervalMesh(Mesh):
    """Mesh point network with equal-interval bin spacing in each dimension."""

    def generate_bin_edges(self) -> NDArray:
        min_vals = np.min(self.reference_coordinates, axis=0)
        max_vals = np.max(self.reference_coordinates, axis=0)
        return np.linspace(min_vals, max_vals, num=self.n_bins + 1).T
