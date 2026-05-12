"""
markov_utils.py
───────────────
Funções utilitárias que complementam os módulos principais
(markov_generators, linear_algebra, others).

Contém apenas lógica — nenhum código de interface.
"""

import numpy as np
from linear_algebra import build_steady_state_system, qr_decomposition, solver_qr


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Normaliza cada linha da matriz para que some 1 (matriz estocástica).
    Linhas de zeros são substituídas por distribuição uniforme.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    n = matrix.shape[0]
    # Linhas zeradas viram uniformes
    zero_rows = (row_sums.flatten() == 0)
    row_sums[zero_rows] = 1.0
    normalized = matrix / row_sums
    for i in np.where(zero_rows)[0]:
        normalized[i] = 1.0 / n
    return normalized


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normaliza vetor de probabilidade; se nulo, retorna uniforme."""
    v = np.clip(v, 0, None)
    s = v.sum()
    if s == 0:
        return np.ones(len(v)) / len(v)
    return v / s


def get_stationary_distribution(P: np.ndarray) -> np.ndarray | None:
    """
    Calcula a distribuição estacionária π via decomposição QR
    (usa build_steady_state_system + qr_decomposition + solver_qr
    de linear_algebra.py).

    Retorna None se o sistema não tiver solução única.
    """
    try:
        A, b = build_steady_state_system(P)
        Q, R = qr_decomposition(A)
        pi = solver_qr(Q, R, b)
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi
    except Exception:
        return None
