import warnings

import numpy as np
import pytest
import scipy

import pytensor.sparse as ps
import pytensor.tensor as pt
from tests.link.numba.sparse.test_basic import compare_numba_and_py_sparse


pytestmark = pytest.mark.filterwarnings("error")

DOT_SHAPES = [((20, 11), (11, 4)), ((10, 3), (3, 1)), ((1, 10), (10, 5))]


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
def test_true_dot_sparse_sparse(x_format, y_format):
    x = ps.matrix(x_format, name="x")
    y = ps.matrix(y_format, name="y")
    rng = np.random.default_rng(208)
    x_values = rng.integers(-2, 3, size=(13, 9))
    y_values = rng.integers(-2, 3, size=(9, 7))
    x_values[rng.random(x_values.shape) < 0.7] = 0
    y_values[rng.random(y_values.shape) < 0.7] = 0
    x_test = scipy.sparse.csr_matrix(x_values.astype("float64")).asformat(x_format)
    y_test = scipy.sparse.csr_matrix(y_values.astype("float64")).asformat(y_format)

    compare_numba_and_py_sparse([x, y], ps.true_dot(x, y), [x_test, y_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("n_cols", [1, 7])
@pytest.mark.parametrize("dtype", ["float64", "int32", "complex64"])
def test_true_dot_sparse_dense(format, n_cols, dtype):
    x = ps.matrix(format, name="x", dtype=dtype)
    y = pt.matrix("y", shape=(9, n_cols), dtype=dtype)
    rng = np.random.default_rng(209)
    x_values = rng.integers(-2, 3, size=(13, 9))
    x_values[rng.random(x_values.shape) < 0.7] = 0
    x_test = scipy.sparse.csr_matrix(x_values.astype(dtype)).asformat(format)
    y_test = rng.integers(-2, 3, size=(9, n_cols)).astype(dtype)

    compare_numba_and_py_sparse([x, y], ps.true_dot(x, y), [x_test, y_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_true_dot_dense_sparse(format):
    x = pt.matrix("x")
    y = ps.matrix(format, name="y")
    rng = np.random.default_rng(210)
    x_test = rng.integers(-2, 3, size=(13, 9)).astype("float64")
    y_values = rng.integers(-2, 3, size=(9, 7))
    y_values[rng.random(y_values.shape) < 0.7] = 0
    y_test = scipy.sparse.csr_matrix(y_values.astype("float64")).asformat(format)

    compare_numba_and_py_sparse([x, y], ps.true_dot(x, y), [x_test, y_test])


class TestComparisons:
    def _comparison_values(self, format):
        x_dense = np.zeros((11, 7))
        x_dense[0, 1] = 2
        x_dense[3, 4] = -1
        x_dense[7, 2] = 3
        x_test = scipy.sparse.csr_matrix(x_dense).asformat(format)

        y_dense = np.zeros((11, 7))
        y_dense[0, 1] = 2
        y_dense[5, 0] = 4
        y_dense[7, 2] = -2
        y_sparse_test = scipy.sparse.csr_matrix(y_dense).asformat(format)

        rng = np.random.default_rng(155)
        y_dense_test = rng.integers(-2, 3, size=(11, 7)).astype("float64")
        y_dense_test[0, 1] = 2
        return x_test, y_sparse_test, y_dense_test

    @pytest.mark.parametrize("comparison", ["eq", "neq", "lt", "gt", "le", "ge"])
    @pytest.mark.parametrize("format", ["csr", "csc"])
    def test_sparse_sparse_comparison(self, comparison, format):
        x = ps.matrix(format, name="x")
        y = ps.matrix(format, name="y")
        x_test, y_test, _ = self._comparison_values(format)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Comparing two sparse matrices using",
                category=scipy.sparse.SparseEfficiencyWarning,
            )
            compare_numba_and_py_sparse(
                [x, y], getattr(ps, comparison)(x, y), [x_test, y_test]
            )

    @pytest.mark.parametrize("comparison", ["eq", "neq", "lt", "gt", "le", "ge"])
    @pytest.mark.parametrize("format", ["csr", "csc"])
    def test_sparse_dense_comparison(self, comparison, format):
        x = ps.matrix(format, name="x")
        y = pt.matrix("y")
        x_test, _, y_test = self._comparison_values(format)
        compare_numba_and_py_sparse(
            [x, y], getattr(ps, comparison)(x, y), [x_test, y_test]
        )

    @pytest.mark.parametrize("comparison", ["eq", "neq", "lt", "gt", "le", "ge"])
    @pytest.mark.parametrize("format", ["csr", "csc"])
    def test_dense_sparse_comparison(self, comparison, format):
        x = pt.matrix("x")
        y = ps.matrix(format, name="y")
        y_test, _, x_test = self._comparison_values(format)
        compare_numba_and_py_sparse(
            [x, y], getattr(ps, comparison)(x, y), [x_test, y_test]
        )

    def _noncanonical_comparison_values(self, format):
        constructor = (
            scipy.sparse.csr_matrix if format == "csr" else scipy.sparse.csc_matrix
        )
        x_test = constructor(
            ([2.0, 3.0, -2.0, 0.0, 4.0], [3, 1, 3, 0, 2], [0, 4, 5] + [5] * 6),
            shape=(8, 8),
        )
        y_sparse_test = constructor(
            ([1.0, -1.0, 2.0, 0.0], [3, 3, 1, 0], [0, 3, 4] + [4] * 6),
            shape=(8, 8),
        )
        y_dense_test = np.zeros((8, 8))
        y_dense_test[0, 1] = 3
        y_dense_test[1, 2] = -1
        return x_test, y_sparse_test, y_dense_test

    @pytest.mark.parametrize("comparison", ["eq", "neq", "lt", "gt", "le", "ge"])
    @pytest.mark.parametrize("format", ["csr", "csc"])
    def test_sparse_sparse_comparison_noncanonical(self, comparison, format):
        x = ps.matrix(format, name="x")
        y = ps.matrix(format, name="y")
        x_test, y_test, _ = self._noncanonical_comparison_values(format)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Comparing two sparse matrices using",
                category=scipy.sparse.SparseEfficiencyWarning,
            )
            compare_numba_and_py_sparse(
                [x, y], getattr(ps, comparison)(x, y), [x_test, y_test]
            )

    @pytest.mark.parametrize("comparison", ["eq", "neq", "lt", "gt", "le", "ge"])
    @pytest.mark.parametrize("format", ["csr", "csc"])
    def test_sparse_dense_comparison_noncanonical(self, comparison, format):
        x = ps.matrix(format, name="x")
        y = pt.matrix("y")
        x_test, _, y_test = self._noncanonical_comparison_values(format)
        compare_numba_and_py_sparse(
            [x, y], getattr(ps, comparison)(x, y), [x_test, y_test]
        )


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "complex64"])
def test_sampling_dot(format, dtype):
    x = pt.matrix("x", dtype=dtype)
    y = pt.matrix("y", dtype=dtype)
    p = ps.matrix(format, name="p", dtype=dtype)
    rng = np.random.default_rng(155)
    x_test = rng.integers(-3, 4, size=(17, 11)).astype(dtype)
    y_test = rng.integers(-3, 4, size=(13, 11)).astype(dtype)
    p_test = scipy.sparse.random(
        17, 13, density=0.2, format=format, dtype=dtype, random_state=rng
    )
    if dtype == "int32":
        p_test.data[:] = rng.integers(-3, 4, size=p_test.nnz)

    compare_numba_and_py_sparse(
        [x, y, p], ps.sampling_dot(x, y, p), [x_test, y_test, p_test]
    )


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sampling_dot_zeros_and_duplicates(format):
    x = pt.matrix("x")
    y = pt.matrix("y")
    p = ps.matrix(format, name="p")
    constructor = (
        scipy.sparse.csr_matrix if format == "csr" else scipy.sparse.csc_matrix
    )
    p_test = constructor(
        ([2.0, -2.0, 0.0, 3.0], [2, 2, 0, 1], [0, 3, 4, 4, 4]),
        shape=(4, 4),
    )
    x_test = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [5.0, 6.0]])
    y_test = np.array([[2.0, 1.0], [1.0, 3.0], [0.0, 0.0], [4.0, 2.0]])

    compare_numba_and_py_sparse(
        [x, y, p], ps.sampling_dot(x, y, p), [x_test, y_test, p_test]
    )


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sampling_dot_negative_strides(format):
    x = pt.matrix("x")
    y = pt.matrix("y")
    p = ps.matrix(format, name="p")
    rng = np.random.default_rng(155)
    x_test = rng.normal(size=(10, 7))[::-1]
    y_test = rng.normal(size=(12, 7))[:, ::-1]
    p_test = scipy.sparse.random(10, 12, density=0.2, format=format, random_state=rng)

    compare_numba_and_py_sparse(
        [x, y, p], ps.sampling_dot(x, y, p), [x_test, y_test, p_test]
    )


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sampling_dot_grad(format):
    x = pt.matrix("x", shape=(9, 7))
    y = pt.matrix("y", shape=(8, 7))
    p = ps.matrix(format, name="p", shape=(9, 8))
    grads = pt.grad(ps.sp_sum(ps.sampling_dot(x, y, p)), [x, y])
    rng = np.random.default_rng(155)
    x_test = rng.normal(size=(9, 7))
    y_test = rng.normal(size=(8, 7))
    p_test = scipy.sparse.random(9, 8, density=0.3, format=format, random_state=rng)

    compare_numba_and_py_sparse([x, y, p], grads, [x_test, y_test, p_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "complex64"])
def test_structured_add_sparse_vector(format, dtype):
    x = ps.matrix(format, name="x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    rng = np.random.default_rng(155)
    x_test = scipy.sparse.random(
        19, 13, density=0.5, format=format, dtype=dtype, random_state=rng
    )
    y_test = rng.normal(size=13).astype(dtype)

    compare_numba_and_py_sparse([x, y], ps.structured_add_s_v(x, y), [x_test, y_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_structured_add_sparse_vector_zeros(format):
    x = ps.matrix(format, name="x")
    y = pt.vector("y")
    constructor = (
        scipy.sparse.csr_matrix if format == "csr" else scipy.sparse.csc_matrix
    )
    x_test = constructor(
        ([2.0, -2.0, 0.0, 3.0, 4.0], [2, 2, 0, 1, 1], [0, 4, 5, 5, 5, 5, 5, 5, 5]),
        shape=(8, 8),
    )
    y_test = np.array([1.0, -3.0, 0.0, 2.0, 1.0, -1.0, 2.0, 3.0])

    compare_numba_and_py_sparse([x, y], ps.structured_add_s_v(x, y), [x_test, y_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_structured_add_sparse_vector_grad(format):
    x = ps.matrix(format, name="x", shape=(19, 13))
    y = pt.vector("y", shape=(13,))
    grads = pt.grad(ps.sp_sum(ps.structured_add_s_v(x, y)), [x, y])
    rng = np.random.default_rng(155)
    x_test = scipy.sparse.random(19, 13, density=0.5, format=format, random_state=rng)
    y_test = rng.normal(size=13)

    compare_numba_and_py_sparse([x, y], grads, [x_test, y_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("dtype", ["float64", "int32", "complex64"])
def test_sparse_sparse_add_data(format, dtype):
    x = ps.matrix(format, name="x", dtype=dtype)
    y = ps.matrix(format, name="y", dtype=dtype)
    constructor = (
        scipy.sparse.csr_matrix if format == "csr" else scipy.sparse.csc_matrix
    )
    x_test = constructor(
        (np.array([2, 3, 4], dtype=dtype), [2, 0, 1], [0, 2, 3, 3]),
        shape=(3, 3),
    )
    y_test = x_test.copy()
    y_test.data[:] = [-2, 5, -4]

    compare_numba_and_py_sparse([x, y], ps.add_s_s_data(x, y), [x_test, y_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sparse_sparse_add_data_grad(format):
    x = ps.matrix(format, name="x", shape=(4, 6))
    y = ps.matrix(format, name="y", shape=(4, 6))
    grads = pt.grad(ps.sp_sum(ps.add_s_s_data(x, y)), [x, y])
    rng = np.random.default_rng(155)
    x_test = scipy.sparse.random(4, 6, density=0.5, format=format, random_state=rng)
    y_test = x_test.copy()

    compare_numba_and_py_sparse([x, y], grads, [x_test, y_test])


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
@pytest.mark.parametrize(
    "x_dtype, y_dtype",
    [
        ("float64", "float64"),
        ("int32", "float32"),
        ("float32", "complex64"),
    ],
)
def test_sparse_sparse_add(x_format, y_format, x_dtype, y_dtype):
    x = ps.matrix(x_format, name="x", dtype=x_dtype)
    y = ps.matrix(y_format, name="y", dtype=y_dtype)
    rng = np.random.default_rng(155)
    x_test = scipy.sparse.random(
        9, 4, density=0.5, format=x_format, dtype=x_dtype, random_state=rng
    )
    y_test = scipy.sparse.random(
        9, 4, density=0.5, format=y_format, dtype=y_dtype, random_state=rng
    )

    compare_numba_and_py_sparse([x, y], x + y, [x_test, y_test])


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
def test_sparse_sparse_add_cancellation(x_format, y_format):
    x = ps.matrix(x_format, name="x")
    y = ps.matrix(y_format, name="y")
    x_constructor = (
        scipy.sparse.csr_matrix if x_format == "csr" else scipy.sparse.csc_matrix
    )
    y_constructor = (
        scipy.sparse.csr_matrix if y_format == "csr" else scipy.sparse.csc_matrix
    )
    x_test = x_constructor(
        ([1.0, 2.0, 3.0, 4.0], [2, 0, 2, 1], [0, 3, 4, 4]), shape=(3, 3)
    )
    y_test = y_constructor(([-2.0, -3.0, 6.0], [2, 2, 0], [0, 2, 3, 3]), shape=(3, 3))

    compare_numba_and_py_sparse([x, y], x + y, [x_test, y_test])


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
def test_sparse_sparse_add_grad(x_format, y_format):
    x = ps.matrix(x_format, name="x", shape=(9, 4))
    y = ps.matrix(y_format, name="y", shape=(9, 4))
    grads = pt.grad(ps.sp_sum(x + y), [x, y])
    rng = np.random.default_rng(155)
    x_test = scipy.sparse.random(9, 4, density=0.5, format=x_format, random_state=rng)
    y_test = scipy.sparse.random(9, 4, density=0.5, format=y_format, random_state=rng)

    compare_numba_and_py_sparse([x, y], grads, [x_test, y_test])


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
@pytest.mark.parametrize(
    "x_dtype, y_dtype",
    [
        ("float64", "float64"),
        ("int32", "float32"),
        ("float32", "complex64"),
    ],
)
def test_sparse_sparse_multiply(x_format, y_format, x_dtype, y_dtype):
    x = ps.matrix(x_format, name="x", dtype=x_dtype)
    y = ps.matrix(y_format, name="y", dtype=y_dtype)
    rng = np.random.default_rng(155)
    x_test = scipy.sparse.random(
        7, 5, density=0.5, format=x_format, dtype=x_dtype, random_state=rng
    )
    y_test = scipy.sparse.random(
        7, 5, density=0.5, format=y_format, dtype=y_dtype, random_state=rng
    )

    compare_numba_and_py_sparse([x, y], x * y, [x_test, y_test])


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
def test_sparse_sparse_multiply_grad(x_format, y_format):
    x = ps.matrix(x_format, name="x", dtype="float64", shape=(7, 5))
    y = ps.matrix(y_format, name="y", dtype="float64", shape=(7, 5))
    grads = pt.grad(ps.sp_sum(x * y), [x, y])
    rng = np.random.default_rng(155)
    x_test = scipy.sparse.random(7, 5, density=0.5, format=x_format, random_state=rng)
    y_test = scipy.sparse.random(7, 5, density=0.5, format=y_format, random_state=rng)
    compare_numba_and_py_sparse([x, y], grads, [x_test, y_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("y_ndim", [0, 1, 2])
def test_sparse_dense_multiply(y_ndim, format):
    x = ps.matrix(format, name="x", shape=(3, 3))
    y = pt.tensor("y", shape=(3,) * y_ndim)
    z = x * y

    rng = np.random.default_rng((155, y_ndim, format == "csr"))
    x_test = scipy.sparse.random(3, 3, density=0.5, format=format, random_state=rng)
    y_test = rng.normal(size=(3,) * y_ndim)

    compare_numba_and_py_sparse(
        [x, y],
        z,
        [x_test, y_test],
    )


@pytest.mark.parametrize("op", [ps.dot, ps.structured_dot])
@pytest.mark.parametrize("sp_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_dot_sparse_dense(op, sp_format, x_shape, y_shape):
    x = ps.matrix(format=sp_format, name="x", shape=x_shape)
    y = pt.matrix("y", shape=y_shape)
    z = op(x, y)

    rng = np.random.default_rng(sum(map(ord, sp_format)) + sum(x_shape) + sum(y_shape))
    x_test = scipy.sparse.random(
        *x_shape, density=0.5, format=sp_format, random_state=rng
    )
    y_test = rng.normal(size=y_shape)

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("op", [ps.dot, ps.structured_dot])
@pytest.mark.parametrize("sp_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_dot_dense_sparse(op, sp_format, x_shape, y_shape):
    x = pt.matrix(name="x", shape=x_shape)
    y = ps.matrix(format=sp_format, name="y", shape=y_shape)
    z = op(x, y)

    rng = np.random.default_rng(sum(map(ord, sp_format)) + sum(x_shape) + sum(y_shape))
    x_test = rng.normal(size=x_shape)
    y_test = scipy.sparse.random(
        *y_shape, density=0.5, format=sp_format, random_state=rng
    )

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("op", [ps.dot, ps.structured_dot])
@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_sparse_dot_sparse_sparse(op, x_format, y_format, x_shape, y_shape):
    x = ps.matrix(x_format, name="x", shape=x_shape)
    y = ps.matrix(y_format, name="y", shape=y_shape)
    z = op(x, y)

    rng = np.random.default_rng(sum(map(ord, x_format)) + sum(map(ord, y_format)))
    x_test = scipy.sparse.random(
        *x_shape, density=0.5, format=x_format, random_state=rng
    )
    y_test = scipy.sparse.random(
        *y_shape, density=0.5, format=y_format, random_state=rng
    )

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("sp_format", ["csr", "csc"])
def test_sparse_spmv(sp_format):
    x = ps.matrix(format=sp_format, name="x", shape=(20, 6))
    y = pt.vector("y", shape=(6,))
    z = ps.dot(x, y)

    rng = np.random.default_rng(sp_format == "csr")
    x_test = scipy.sparse.random(20, 6, density=0.5, format=sp_format, random_state=rng)
    y_test = rng.normal(size=(6,))

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize(
    "x_dtype, y_dtype",
    [
        ("int64", "complex64"),
        ("int64", "float32"),
    ],
)
def test_structured_dot_upcast(x_dtype, y_dtype):
    """Numba scalar-array mul keeps the array dtype; numpy upcasts to a wider type."""
    x = ps.matrix(format="csc", name="x", dtype=x_dtype, shape=(4, 3))
    y = pt.matrix("y", dtype=y_dtype, shape=(3, 5))
    z = ps.structured_dot(x, y)

    x_test = scipy.sparse.csc_matrix(
        np.array([[97, 0, 0], [0, 83, 0], [0, 0, 71], [42, 0, 0]], dtype=x_dtype)
    )
    y_test = np.array(
        [
            [9.12345, -3.98765, 7.55555, 1.23456, -5.67890],
            [2.34567, 8.76543, -4.32109, 6.54321, 0.98765],
            [-1.11111, 3.33333, 9.99999, -7.77777, 2.22222],
        ],
        dtype=y_dtype,
    )

    def strict_assert(a, b):
        if scipy.sparse.issparse(a):
            a = a.toarray()
        if scipy.sparse.issparse(b):
            b = b.toarray()
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=0, strict=True)

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test], assert_fn=strict_assert)


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc", "dense"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_structured_dot_grad(x_format, y_format, x_shape, y_shape):
    rng = np.random.default_rng()
    g_xy_shape = (x_shape[0], y_shape[1])

    x = ps.matrix(format=x_format, name="x", shape=x_shape)
    x_test = scipy.sparse.random(*x_shape, density=0.4, format=x_format)

    if y_format == "dense":
        y = pt.matrix("y", shape=y_shape)
        g_xy = pt.matrix(name="g_xy", shape=g_xy_shape)
        y_test = rng.normal(size=y_shape)
        g_xy_test = rng.normal(size=g_xy_shape)
    else:
        y = ps.matrix(format=y_format, name="y", shape=y_shape)
        g_xy = ps.matrix(format=x_format, name="g_xy", shape=g_xy_shape)
        y_test = scipy.sparse.random(*y_shape, density=0.5, format=y_format)
        g_xy_test = scipy.sparse.random(*g_xy_shape, density=0.3, format=x_format)

    z = ps.structured_dot_grad(x, y, g_xy)
    compare_numba_and_py_sparse([x, y, g_xy], z, [x_test, y_test, g_xy_test])


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sparse_sum(format, axis):
    x = ps.matrix(format=format, name="x", shape=(7, 5))
    z = ps.sp_sum(x, axis=axis)
    x_test = scipy.sparse.random(7, 5, density=0.4, format=format)

    compare_numba_and_py_sparse([x], z, [x_test])
