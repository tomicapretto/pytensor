import numpy as np
import scipy.sparse as sp

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.string_codegen import CODE_TOKEN, build_source_code
from pytensor.sparse.linalg import SparseBlockDiagonal


def _canonicalize_block_diagonal(data, indices, indptr):
    nnz = 0
    old_start = 0
    for major in range(len(indptr) - 1):
        old_end = indptr[major + 1]
        sorted_unique = True
        for k in range(old_start + 1, old_end):
            if indices[k] <= indices[k - 1]:
                sorted_unique = False
                break

        if sorted_unique:
            for k in range(old_start, old_end):
                data[nnz] = data[k]
                indices[nnz] = indices[k]
                nnz += 1
        else:
            row_indices = indices[old_start:old_end].copy()
            row_data = data[old_start:old_end].copy()
            for k in np.argsort(row_indices):
                index = row_indices[k]
                if nnz > indptr[major] and indices[nnz - 1] == index:
                    data[nnz - 1] += row_data[k]
                else:
                    indices[nnz] = index
                    data[nnz] = row_data[k]
                    nnz += 1

        old_start = old_end
        indptr[major + 1] = nnz

    return data[:nnz], indices[:nnz], indptr


@register_funcify_default_op_cache_key(SparseBlockDiagonal)
def numba_funcify_SparseBlockDiagonal(op, node, **kwargs):
    format = op.format
    dtype = node.outputs[0].type.dtype
    names = [f"arr{i}" for i in range(len(node.inputs))]
    block_names = [f"block{i}" for i in range(len(node.inputs))]
    n_major_axis = 0 if format == "csr" else 1
    n_minor_axis = 1 - n_major_axis
    constructor = "csr_matrix" if format == "csr" else "csc_matrix"
    canonicalize = numba_basic.numba_njit(_canonicalize_block_diagonal)

    @numba_basic.numba_njit
    def dense_to_cs_with_explicit_zeros(x):
        n_rows, n_cols = x.shape
        n_major = n_rows if format == "csr" else n_cols
        n_minor = n_cols if format == "csr" else n_rows
        data = np.empty(n_rows * n_cols, dtype=x.dtype)
        indices = np.empty(n_rows * n_cols, dtype=np.uint32)
        indptr = np.empty(n_major + 1, dtype=np.uint32)
        indptr[0] = 0
        for major in range(n_major):
            for minor in range(n_minor):
                k = major * n_minor + minor
                data[k] = x[major, minor] if format == "csr" else x[minor, major]
                indices[k] = minor
            indptr[major + 1] = (major + 1) * n_minor

        components = (data, indices.view(np.int32), indptr.view(np.int32))
        if format == "csr":
            return sp.csr_matrix(components, shape=x.shape)
        return sp.csc_matrix(components, shape=x.shape)

    code = [
        f"def block_diagonal({', '.join(names)}):",
        CODE_TOKEN.INDENT,
    ]
    for name, block, inp in zip(names, block_names, node.inputs, strict=True):
        if getattr(inp.type, "format", None) == format:
            code.append(f"{block} = {name}")
        elif getattr(inp.type, "format", None) is None:
            code.append(f"{block} = dense_to_cs_with_explicit_zeros({name})")
        else:
            code.append(f"{block} = sp.{constructor}({name})")

    code.extend(
        [
            f"n_rows = {' + '.join(f'{block}.shape[0]' for block in block_names)}",
            f"n_cols = {' + '.join(f'{block}.shape[1]' for block in block_names)}",
            f"total_nnz = {' + '.join(f'{block}.indptr[-1]' for block in block_names)}",
            f"data = np.empty(total_nnz, dtype=np.{dtype})",
            "indices = np.empty(total_nnz, dtype=np.uint32)",
            f"indptr = np.empty({'n_rows' if format == 'csr' else 'n_cols'} + 1, dtype=np.uint32)",
            "indptr[0] = 0",
            "major_offset = 0",
            "minor_offset = 0",
            "data_offset = 0",
        ]
    )
    for block in block_names:
        code.extend(
            [
                f"block_nnz = {block}.indptr[-1]",
                f"block_major = {block}.shape[{n_major_axis}]",
                f"block_minor = {block}.shape[{n_minor_axis}]",
                f"data[data_offset:data_offset + block_nnz] = {block}.data[:block_nnz]",
                f"indices[data_offset:data_offset + block_nnz] = {block}.indices.view(np.uint32)[:block_nnz] + minor_offset",
                f"indptr[major_offset + 1:major_offset + block_major + 1] = {block}.indptr.view(np.uint32)[1:] + data_offset",
                "major_offset += block_major",
                "minor_offset += block_minor",
                "data_offset += block_nnz",
            ]
        )
    code.extend(
        [
            "data, indices, indptr = canonicalize(data, indices, indptr)",
            f"return sp.{constructor}((data, indices.view(np.int32), indptr.view(np.int32)), shape=(n_rows, n_cols))",
            CODE_TOKEN.DEDENT,
        ]
    )
    block_diagonal = compile_numba_function_src(
        build_source_code(code),
        "block_diagonal",
        globals()
        | {
            "canonicalize": canonicalize,
            "dense_to_cs_with_explicit_zeros": dense_to_cs_with_explicit_zeros,
        },
    )
    return numba_basic.numba_njit(block_diagonal), 1
