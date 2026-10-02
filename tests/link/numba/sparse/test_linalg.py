import numpy as np
import pytest
import scipy.sparse as sp

import pytensor.sparse as ps
import pytensor.tensor as pt
from pytensor.sparse.linalg import block_diag
from tests.link.numba.sparse.test_basic import compare_numba_and_py_sparse


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sparse_block_diagonal_mixed_inputs(format):
    x = ps.matrix("csr", name="x", dtype="float32")
    y = pt.matrix("y", dtype="int32")
    z = ps.matrix("csc", name="z", dtype="float64")

    x_test = sp.csr_matrix(np.eye(11, 7, dtype="float32"))
    y_test = np.arange(45, dtype="int32").reshape(5, 9) % 4 - 1
    z_test = sp.csc_matrix(np.eye(8, 4, dtype="float64") * 2)

    compare_numba_and_py_sparse(
        [x, y, z],
        block_diag(x, y, z, format=format),
        [x_test, y_test, z_test],
    )


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sparse_block_diagonal_noncanonical(format):
    x = ps.matrix("csr", name="x")
    y = ps.matrix("csc", name="y")
    x_test = sp.csr_matrix(
        ([2.0, 3.0, -2.0, 0.0], [3, 1, 3, 0], [0, 4] + [4] * 7),
        shape=(8, 7),
    )
    y_test = sp.csc_matrix(
        ([1.0, -1.0, 0.0, 2.0], [2, 2, 0, 1], [0, 3, 4] + [4] * 5),
        shape=(6, 7),
    )

    compare_numba_and_py_sparse(
        [x, y], block_diag(x, y, format=format), [x_test, y_test]
    )


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sparse_block_diagonal_empty_dimensions(format):
    x = ps.matrix("csr", name="x")
    y = pt.matrix("y")
    z = ps.matrix("csc", name="z")
    x_test = sp.csr_matrix((0, 5))
    y_test = np.empty((4, 0))
    z_test = sp.csc_matrix(np.eye(2, 3))

    compare_numba_and_py_sparse(
        [x, y, z],
        block_diag(x, y, z, format=format),
        [x_test, y_test, z_test],
    )
