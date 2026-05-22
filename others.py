import numpy as np
from linear_algebra import build_steady_state_system, qr_decomposition, solver_qr

import networkx as nx


def simulate_trajectory(P: np.ndarray, pi0: np.ndarray, size: int)-> np.ndarray:
    """
    Simula uma trajetória de tamanho "size" onde o estado inicial é escolhido
    baseado em uma distribuição de probabilidade inicial pi_0.
    """

    n = P.shape[0]

    current_state = np.random.choice(n, p=pi0)
    trajectory = [current_state]


    for _ in range(size):
        next_state = np.random.choice(n, p=P[current_state])
        trajectory.append(next_state)
        current_state = next_state
        
    return trajectory

def diagnose_with_graphs(P: np.ndarray) -> tuple[bool, bool, int]:
    """
    Verifica se a cadeia de markov é aperiódica e irredutível e
    o número de classes comunicantes

    por meio da matriz de transição P

    Args:
    - P : matriz de transição nxn

    return:
        (is_irreducible, is_aperiodic, n_componentes)
    """
    
    # Grafo associado a P
    G = nx.from_numpy_array(P, create_using=nx.DiGraph)
        
    # Irredutibilidade (Fortemente Conexo)
    is_irreducible = nx.is_strongly_connected(G)
    
    # Aperiodicidade
    # Um grafo é aperiódico se o MDC dos comprimentos de todos os ciclos é 1
    if is_irreducible:
        is_aperiodic = nx.is_aperiodic(G)
    else:
        # Se não for irredutível, a aperiodicidade deve ser checada por componente
        is_aperiodic = all(nx.is_aperiodic(G.subgraph(c)) 
                          for c in nx.strongly_connected_components(G))

    # Análise de Componentes
    scc = list(nx.strongly_connected_components(G))

    return is_irreducible, is_aperiodic, scc


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
    """
    Normaliza vetor de probabilidade; se nulo, retorna uniforme.
    """
    
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

        return pi
        
    except Exception:
        return None

def resize_chain(
    P: np.ndarray,
    pi0: np.ndarray,
    new_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona P (nxn) e pi0 (n,) para new_n estados, preservando
    a estrutura existente.

    Encolher (new_n < old_n):
        Remove as últimas linhas e colunas (últimos estados).

    Crescer (new_n > old_n):
        Embute a matriz antiga no bloco superior-esquerdo.
        As novas linhas e colunas em P são inicializadas com zeros.
        O vetor pi0 é estendido com zeros para os novos estados.
    """
    old_n = P.shape[0]
    if new_n == old_n:
        return P.copy(), pi0.copy()

    if new_n < old_n:
        P_new   = P[:new_n, :new_n].copy()
        pi0_new = pi0[:new_n].copy()

    else:
        P_new = np.zeros((new_n, new_n))
        pi0_new = np.zeros(new_n)
        
        P_new[:old_n, :old_n] = P
        pi0_new[:old_n] = pi0

    return P_new, pi0_new


def resize_state_names(names: list[str], new_n: int) -> list[str]:
    """
    Mantém os nomes existentes ao encolher; acrescenta S{i} ao crescer.
    """
    old_n = len(names)
    if new_n <= old_n:
        return names[:new_n]
    return names + [f"S{i}" for i in range(old_n, new_n)]