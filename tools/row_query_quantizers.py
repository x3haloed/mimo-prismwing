"""Deterministic row-query quantizer primitives for PW-0302.

These NumPy references favor auditable semantics over packed-kernel speed.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


NORMAL = NormalDist()


def lloyd_max_normal_centroids(bits: int, iterations: int = 200) -> np.ndarray:
    """Return symmetric Lloyd-Max centroids for a standard normal variable."""
    if bits < 1 or bits > 8 or iterations < 1:
        raise ValueError("unsupported Lloyd-Max configuration")
    count = 1 << bits
    centroids = np.asarray(
        [NORMAL.inv_cdf((index + 0.5) / count) for index in range(count)],
        dtype=np.float64,
    )
    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        updated = np.empty_like(centroids)
        for index in range(count):
            lower = -math.inf if index == 0 else float(boundaries[index - 1])
            upper = math.inf if index == count - 1 else float(boundaries[index])
            probability = NORMAL.cdf(upper) - NORMAL.cdf(lower)
            lower_density = 0.0 if lower == -math.inf else NORMAL.pdf(lower)
            upper_density = 0.0 if upper == math.inf else NORMAL.pdf(upper)
            updated[index] = (lower_density - upper_density) / probability
        if np.max(np.abs(updated - centroids)) < 1e-14:
            centroids = updated
            break
        centroids = updated
    if not np.all(np.isfinite(centroids)) or not np.all(np.diff(centroids) > 0):
        raise ValueError("invalid Lloyd-Max solution")
    return centroids


def seeded_signs(dimension: int, seed: int) -> np.ndarray:
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("dimension must be a positive power of two")
    generator = np.random.Generator(np.random.PCG64(seed))
    return np.where(generator.integers(0, 2, size=dimension, dtype=np.uint8), 1.0, -1.0)


def fwht_rows(values: np.ndarray) -> np.ndarray:
    """Apply an orthonormal Walsh-Hadamard transform to the final axis."""
    result = np.asarray(values, dtype=np.float64).copy()
    dimension = result.shape[-1]
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("final dimension must be a positive power of two")
    width = 1
    while width < dimension:
        view = result.reshape(*result.shape[:-1], -1, width * 2)
        left = view[..., :width].copy()
        right = view[..., width:].copy()
        view[..., :width] = left + right
        view[..., width:] = left - right
        width *= 2
    result /= math.sqrt(dimension)
    return result


def rotate_rows(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    if values.shape[-1] != signs.shape[0]:
        raise ValueError("rotation dimension mismatch")
    return fwht_rows(values * signs)


def inverse_rotate_rows(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    if values.shape[-1] != signs.shape[0]:
        raise ValueError("inverse rotation dimension mismatch")
    return fwht_rows(values) * signs


def quantize_rotated_rows(
    rows: np.ndarray,
    bits: int,
    signs: np.ndarray,
    norm_dtype: np.dtype = np.dtype("float32"),
) -> dict[str, np.ndarray]:
    """Quantize unit-normalized rotated rows using the normal Lloyd-Max grid."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or not np.all(np.isfinite(rows)):
        raise ValueError("rows must be a finite matrix")
    dimension = rows.shape[1]
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero row cannot be normalized")
    standardized_centroids = lloyd_max_normal_centroids(bits)
    unit_centroids = standardized_centroids / math.sqrt(dimension)
    rotated_unit = rotate_rows(rows / norms[:, None], signs)
    boundaries = (unit_centroids[:-1] + unit_centroids[1:]) * 0.5
    codes = np.searchsorted(boundaries, rotated_unit).astype(np.uint8)
    stored_norms = norms.astype(norm_dtype)
    reconstructed = unit_centroids[codes] * stored_norms.astype(np.float64)[:, None]
    return {
        "codes": codes,
        "norms": stored_norms,
        "unit_centroids": unit_centroids,
        "reconstructed_rotated": reconstructed,
        "residual": rows - inverse_rotate_rows(reconstructed, signs),
    }


def estimate_mse_inner_products(queries: np.ndarray, quantized: dict, signs: np.ndarray) -> np.ndarray:
    rotated_queries = rotate_rows(queries, signs)
    return rotated_queries @ quantized["reconstructed_rotated"].T


def turbo_prod_quantize_rows(rows: np.ndarray, signs: np.ndarray, qjl_signs: np.ndarray) -> dict:
    """Build the 5-bit MSE plus 1-bit structured-QJL PW-0302 candidate."""
    base = quantize_rotated_rows(rows, 5, signs, np.dtype("float16"))
    residual = base["residual"]
    residual_norms = np.linalg.norm(residual, axis=1).astype(np.float16)
    normalized = residual / residual_norms.astype(np.float64)[:, None]
    qjl_codes = np.sign(rotate_rows(normalized, qjl_signs))
    qjl_codes[qjl_codes == 0] = 1
    return {"base": base, "residual_norms": residual_norms, "qjl_codes": qjl_codes}


def estimate_turbo_prod_inner_products(
    queries: np.ndarray,
    quantized: dict,
    signs: np.ndarray,
    qjl_signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, MSE-base, and structured-QJL correction estimates."""
    queries = np.asarray(queries, dtype=np.float64)
    base = estimate_mse_inner_products(queries, quantized["base"], signs)
    dimension = queries.shape[1]
    qjl_queries = rotate_rows(queries, qjl_signs)
    correction = (
        math.sqrt(math.pi / 2.0) / math.sqrt(dimension)
        * (qjl_queries @ quantized["qjl_codes"].T)
        * quantized["residual_norms"].astype(np.float64)[None, :]
    )
    return base + correction, base, correction


def fit_block_covariance_transform(train_queries: np.ndarray, block: int = 128) -> dict:
    train_queries = np.asarray(train_queries, dtype=np.float64)
    if train_queries.ndim != 2 or train_queries.shape[1] % block:
        raise ValueError("invalid block-covariance training matrix")
    bases = []
    roots = []
    for start in range(0, train_queries.shape[1], block):
        values = train_queries[:, start : start + block]
        covariance = values.T @ values / max(1, values.shape[0])
        eigenvalues, basis = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        basis = basis[:, order]
        floor = max(float(np.mean(eigenvalues)) * 0.001, 1e-12)
        bases.append(basis)
        roots.append(np.sqrt(np.maximum(eigenvalues, floor)))
    return {"block": block, "bases": np.stack(bases), "roots": np.stack(roots)}


def covariance_transform_queries(queries: np.ndarray, transform: dict) -> np.ndarray:
    queries = np.asarray(queries, dtype=np.float64)
    block = transform["block"]
    output = np.empty_like(queries)
    for index, start in enumerate(range(0, queries.shape[1], block)):
        output[:, start : start + block] = (
            queries[:, start : start + block] @ transform["bases"][index]
        )
    return output


def covariance_transform_rows(rows: np.ndarray, transform: dict) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    block = transform["block"]
    output = np.empty_like(rows)
    for index, start in enumerate(range(0, rows.shape[1], block)):
        output[:, start : start + block] = (
            rows[:, start : start + block] @ transform["bases"][index]
        )
    return output


def quantize_covariance_rows(rows: np.ndarray, transform: dict, bits: int = 6) -> dict:
    transformed = covariance_transform_rows(rows, transform)
    dimension = transformed.shape[1]
    norms = np.linalg.norm(transformed, axis=1)
    centroids = lloyd_max_normal_centroids(bits) / math.sqrt(dimension)
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    codes = np.searchsorted(boundaries, transformed / norms[:, None]).astype(np.uint8)
    stored_norms = norms.astype(np.float32)
    return {
        "codes": codes,
        "norms": stored_norms,
        "centroids": centroids,
        "reconstructed_transformed": centroids[codes] * stored_norms[:, None],
    }


def estimate_covariance_inner_products(queries: np.ndarray, quantized: dict, transform: dict) -> np.ndarray:
    return covariance_transform_queries(queries, transform) @ quantized["reconstructed_transformed"].T


def lloyd_empirical_centroids(values: np.ndarray, levels: int = 64, iterations: int = 40) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)) or values.size < levels:
        raise ValueError("invalid empirical Lloyd-Max population")
    centroids = np.quantile(values, (np.arange(levels) + 0.5) / levels)
    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        codes = np.searchsorted(boundaries, values)
        counts = np.bincount(codes, minlength=levels)
        sums = np.bincount(codes, weights=values, minlength=levels)
        updated = centroids.copy()
        present = counts > 0
        updated[present] = sums[present] / counts[present]
        if np.max(np.abs(updated - centroids)) < 1e-12:
            centroids = updated
            break
        centroids = updated
    if not np.all(np.diff(centroids) > 0):
        raise ValueError("degenerate empirical codebook")
    return centroids


def quantize_covariance_shared_block_codebooks(
    rows: np.ndarray, transform: dict, levels: int = 64
) -> dict:
    """Use one learned scalar grid per transformed channel block, shared by all rows."""
    transformed = covariance_transform_rows(rows, transform)
    block = transform["block"]
    reconstructed = np.empty_like(transformed)
    codebooks = []
    for index, start in enumerate(range(0, transformed.shape[1], block)):
        values = transformed[:, start : start + block]
        centroids = lloyd_empirical_centroids(values, levels=levels)
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        codes = np.searchsorted(boundaries, values)
        reconstructed[:, start : start + block] = centroids[codes]
        codebooks.append(centroids)
    return {"reconstructed_transformed": reconstructed, "codebooks": np.stack(codebooks)}
