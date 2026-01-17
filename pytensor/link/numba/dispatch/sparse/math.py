import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.sparse import SparseDenseMultiply, SparseDenseVectorMultiply
from pytensor.sparse import Dot, StructuredDot


@register_funcify_default_op_cache_key(SparseDenseMultiply)
@register_funcify_default_op_cache_key(SparseDenseVectorMultiply)
def numba_funcify_SparseDenseMultiply(op, node, **kwargs):
    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype
    format = z.type.format
    same_dtype = x.type.dtype == out_dtype

    if y.ndim == 0:

        @numba_basic.numba_njit
        def sparse_multiply_scalar(x, y):
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)
            # Numba doesn't know how to handle in-place mutation / assignment of fields
            # z.data *= y
            z_data = z.data
            z_data *= y
            return z

        return sparse_multiply_scalar

    elif y.ndim == 1:

        @numba_basic.numba_njit
        def sparse_dense_multiply(x, y):
            assert x.shape[1] == y.shape[0]
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)

            M, N = x.shape
            indices = x.indices
            indptr = x.indptr
            z_data = z.data
            if format == "csc":
                for j in range(0, N):
                    for i_idx in range(indptr[j], indptr[j + 1]):
                        z_data[i_idx] *= y[j]
                return z

            else:
                for i in range(0, M):
                    for j_idx in range(indptr[i], indptr[i + 1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[j]

            return z

        return sparse_dense_multiply

    else:  # y.ndim == 2

        @numba_basic.numba_njit
        def sparse_dense_multiply(x, y):
            assert x.shape == y.shape
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)

            M, N = x.shape
            indices = x.indices
            indptr = x.indptr
            z_data = z.data

            if format == "csc":
                for j in range(0, N):
                    for i_idx in range(indptr[j], indptr[j + 1]):
                        i = indices[i_idx]
                        z_data[i_idx] *= y[i, j]
                return z

            else:
                for i in range(0, M):
                    for j_idx in range(indptr[i], indptr[i + 1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[i, j]
            # breakpoint()
            return z

        return sparse_dense_multiply


@register_funcify_default_op_cache_key(Dot)
def numba_funcify_SparseDenseDot(op, node, **kwargs):
    x, y = node.inputs
    [z] = node.outputs
    sparse_format = x.type.format
    out_dtype = z.type.dtype

    # (n, p) @ (p, k) -> (n, k)
    @numba_basic.numba_njit
    def sparse_dense_dot(x, y):
        assert x.shape[1] == y.shape[0]
        n = x.shape[0]
        k = y.shape[1]
        z = np.zeros((n, k), dtype=out_dtype)

        indices = x.indices
        indptr = x.indptr
        data = x.data

        if sparse_format == "csc":
            p = x.shape[1]
            for col_idx in range(p):
                col_start = indptr[col_idx]
                col_end = indptr[col_idx + 1]

                for idx in range(col_start, col_end):
                    row_idx = indices[idx]
                    value = data[idx]
                    z[row_idx, :] += value * y[col_idx, :]
        else:
            for row_idx in range(n):
                row_start = indptr[row_idx]
                row_end = indptr[row_idx + 1]
                for idx in range(row_start, row_end):
                    col_idx = indices[idx]
                    value = data[idx]
                    z[row_idx, :] += value * y[col_idx, :]
        return z

    return sparse_dense_dot
