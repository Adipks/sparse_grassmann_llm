from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from models.blocks import MaskedLinear


def build_2to4_mask(weight: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Build a binary 2:4 structured sparsity mask for `weight`.

    For every contiguous block of 4 elements along `dim`, exactly 2 entries
    are kept (set to 1) and 2 are zeroed (set to 0). Any leftover tail that
    is not divisible by 4 is left dense (all ones).
    """
    if dim < 0:
        dim = weight.dim() + dim

    mask = torch.zeros_like(weight, device=weight.device, dtype=weight.dtype)
    size_along = weight.shape[dim]
    n_blocks = size_along // 4
    tail = size_along % 4

    # Move the target dimension to the last axis for easier slicing.
    perm = list(range(weight.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    inv_perm = [0] * weight.dim()
    for i, p in enumerate(perm):
        inv_perm[p] = i

    w_view = weight.permute(*perm)
    m_view = mask.permute(*perm)

    leading_shape = w_view.shape[:-1]
    flat_leading = int(math.prod(leading_shape)) if leading_shape else 1
    w_flat = w_view.contiguous().view(flat_leading, size_along)
    m_flat = m_view.contiguous().view(flat_leading, size_along)

    for row in range(flat_leading):
        for b in range(n_blocks):
            start = b * 4
            end = start + 4
            # Randomly choose which 2 entries to keep.
            idx = torch.randperm(4, device=weight.device)[:2]
            m_flat[row, start + idx] = 1.0
        if tail:
            # Leave the tail dense.
            m_flat[row, n_blocks * 4 :] = 1.0

    # Restore original layout.
    m_view = m_flat.view(*leading_shape, size_along)
    return m_view.permute(*inv_perm).to(weight.device)


def apply_2to4_masks(model: nn.Module, dims: Iterable[int] = (1,)) -> None:
    """
    Apply 2:4 masks to all MaskedLinear layers in `model`.

    `dims` specifies candidate dimensions along which to enforce the 2:4
    pattern; by default we use the input dimension (dim=1 for [out, in]).
    """
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            with torch.no_grad():
                weight = module.weight
                # Use the first viable dimension.
                dim = next(iter(dims))
                mask = build_2to4_mask(weight, dim=dim)
                module.set_mask(mask)


def try_convert_to_semi_structured_sparse(model: nn.Module) -> nn.Module:
    """
    Placeholder hook to convert 2:4 masked weights to semi-structured sparse
    kernels where supported by PyTorch / CUDA.

    On PyTorch builds that expose `torch.sparse.to_sparse_semi_structured`,
    this can be used at deployment time to materialize Sparse Tensor Core
    weights. If the API is unavailable, this function is a no-op.
    """
    to_sparse = getattr(torch.sparse, "to_sparse_semi_structured", None)
    if to_sparse is None:
        # No-op: environment does not support semi-structured sparse weights.
        return model

    for module in model.modules():
        if isinstance(module, MaskedLinear):
            with torch.no_grad():
                dense_weight = module.weight * module.weight_mask
                # Best-effort: API signature may vary across PyTorch versions.
                try:
                    sparse_weight, meta = to_sparse(dense_weight)
                except TypeError:
                    # Fallback: keep dense weights if signature mismatched.
                    continue
                module.weight = nn.Parameter(sparse_weight)
                module.register_buffer("sparse_meta", meta, persistent=False)
    return model

