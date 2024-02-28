import numpy as np
import pytest

from synthetic_knn.point_networks import (
    EqualIntervalMesh,
    FuzzedNetwork,
    QuantileMesh,
    ReferenceNetwork,
)


def test_reference_network():
    """Test that the reference network returns the reference coordinates."""
    rng = np.random.default_rng(42)
    reference_coordinates = rng.normal(size=(100, 3))
    assert np.array_equal(
        ReferenceNetwork().network_coordinates(reference_coordinates),
        reference_coordinates,
    )


def test_fuzzed_network():
    """Test that the fuzzed network returns coordinates between minimum_distance
    and maximum_distance of reference coordinates."""
    minimum_distance = 0.1
    maximum_distance = 0.5
    rng = np.random.default_rng(42)
    reference_coordinates = rng.normal(size=(100, 3))
    network = FuzzedNetwork(
        minimum_distance=minimum_distance, maximum_distance=maximum_distance
    )
    assert np.all(
        np.linalg.norm(
            network.network_coordinates(reference_coordinates) - reference_coordinates,
            axis=1,
        )
        >= minimum_distance
    )
    assert np.all(
        np.linalg.norm(
            network.network_coordinates(reference_coordinates) - reference_coordinates,
            axis=1,
        )
        <= maximum_distance
    )


@pytest.mark.parametrize(
    "mesh_network",
    [QuantileMesh, EqualIntervalMesh],
    ids=["quantile", "equal_interval"],
)
@pytest.mark.parametrize("n_bins", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("n_dimensions", [2, 3, 4])
def test_mesh_generation(ndarrays_regression, mesh_network, n_bins, n_dimensions):
    """Test that the mesh network returns the correct mesh coordinates
    held in regressions."""
    rng = np.random.default_rng(42)
    reference_coordinates = rng.normal(size=(100, n_dimensions))
    mesh = mesh_network(n_bins=n_bins)
    ndarrays_regression.check(
        dict(network_coordinates=mesh.network_coordinates(reference_coordinates))
    )


@pytest.mark.parametrize(
    "reference_coordinates_type",
    [np.array, list, tuple],
    ids=["np.array", "list", "tuple"],
)
@pytest.mark.parametrize(
    "network",
    [ReferenceNetwork, FuzzedNetwork, QuantileMesh, EqualIntervalMesh],
    ids=["reference", "fuzzed", "quantile", "equal_interval"],
)
def test_mesh_typing(reference_coordinates_type, network):
    """Test that the mesh network accepts different types of reference coordinates."""
    reference_coordinates = reference_coordinates_type(((1, 2), (3, 4)))
    network().network_coordinates(reference_coordinates)
