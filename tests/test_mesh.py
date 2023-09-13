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
    arr = rng.normal(size=(100, n_dimensions))
    mesh = MeshCoords(arr=arr, n_bins=n_bins, classifier=bin_method)
    ndarrays_regression.check(dict(mesh_coords=mesh.to_coords()))
