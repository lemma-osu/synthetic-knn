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
    """Abstract base class for point networks"""

    def fit(self, reference_coordinates: Sequence | NDArray):
        """Set the reference coordinates for the network"""
        self.reference_coordinates = np.array(reference_coordinates)
        return self

    @abstractmethod
    def network_coordinates(self) -> NDArray:
        """Return the point network coordinates as a n-D array"""
        ...


class ReferenceNetwork(PointNetwork):
    """Point network that uses the reference coordinates as the network coordinates"""

    def network_coordinates(self) -> NDArray:
        return self.reference_coordinates


class FuzzedNetwork(PointNetwork):
    """Point network that 'fuzzes' the reference coordinates by a random distance
    between the minimum_distance and the maximum_distance"""

    def __init__(
        self,
        minimum_distance: float = 0.1,
        maximum_distance: float = 0.5,
    ):
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance

    def _fuzz_point(self, point: NDArray) -> tuple[float, ...]:
        """Generate a fuzzed point in n-D space by adding a random offset
        to the original point"""
        # Generate a random direction vector
        rng = np.random.default_rng()
        direction = rng.standard_normal(len(point))

        # Normalize the direction vector
        direction /= np.linalg.norm(direction)

        # Scale the direction vector by a random distance between
        # min_distance and max_distance
        distance = rng.uniform(self.minimum_distance, self.maximum_distance)
        offset = direction * distance

        # Add the offset to the original point
        return tuple(coord + offset_coord for coord, offset_coord in zip(point, offset))

    def network_coordinates(self) -> NDArray:
        """Generate a network of fuzzed points in n-D space"""
        return np.array(
            [self._fuzz_point(point) for point in self.reference_coordinates]
        )


class Mesh(PointNetwork):
    """Base class for mesh point networks"""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def bin_midpoints(self, bins: list[NDArray]) -> NDArray:
        """Find midpoints of bins in each dimension to serve as mesh points."""
        midpoint_coordinates = np.array([(b[:-1] + b[1:]) / 2.0 for b in bins])
        return np.vstack(
            list(map(np.ravel, np.meshgrid(*midpoint_coordinates, indexing="xy")))
        ).T


class QuantileMesh(Mesh):
    """Mesh point network with quantile bin spacing in each dimension."""

    def network_coordinates(self) -> NDArray:
        q = np.linspace(0.0, 1.0, self.n_bins + 1)
        bins = [
            np.quantile(self.reference_coordinates[:, i], q)
            for i in range(self.reference_coordinates.shape[1])
        ]
        return self.bin_midpoints(bins)


class EqualIntervalMesh(Mesh):
    """Mesh point network with equal-interval bin spacing in each dimension."""

    def network_coordinates(self) -> NDArray:
        bins = [
            np.linspace(
                min(self.reference_coordinates[:, i]),
                max(self.reference_coordinates[:, i]),
                self.n_bins + 1,
            )
            for i in range(self.reference_coordinates.shape[1])
        ]
        return self.bin_midpoints(bins)
