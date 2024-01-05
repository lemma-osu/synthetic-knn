import numpy as np
import pytest

from synthetic_knn.mesh import ClassifierEnum, MeshCoords


@pytest.mark.parametrize(
    "bin_method",
    [ClassifierEnum.QUANTILE, ClassifierEnum.EQUAL_INTERVAL],
    ids=["quantile", "equal_interval"],
)
@pytest.mark.parametrize("n_bins", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("n_dimensions", [2, 3, 4])
def test_mesh_generation(ndarrays_regression, bin_method, n_bins, n_dimensions):
    rng = np.random.default_rng(42)
    reference_coordinates = rng.normal(size=(100, n_dimensions))
    mesh = MeshCoords(reference_coordinates, n_bins=n_bins, classifier=bin_method)
    ndarrays_regression.check(dict(mesh_coords=mesh.to_coords()))


@pytest.mark.parametrize(
    "reference_coordinate_type",
    [np.array, list, tuple],
    ids=["np.array", "list", "tuple"],
)
def test_mesh_typing(reference_coordinate_type):
    reference_coordinates = reference_coordinate_type(((1, 2), (3, 4)))
    MeshCoords(
        reference_coordinates=reference_coordinates,
        n_bins=5,
        classifier=ClassifierEnum.QUANTILE,
    )


def test_mesh_invalid_classifier():
    with pytest.raises(ValueError, match="not a valid ClassifierEnum"):
        MeshCoords(
            reference_coordinates=np.array(((1, 2), (3, 4))),
            n_bins=5,
            classifier=ClassifierEnum(3),
        )
