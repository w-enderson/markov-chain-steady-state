import numpy as np
from markov_generators import generate_transition_matrix, generate_probability_vector
from linear_algebra import build_steady_state_system, qr_decomposition, solver_qr

import networkx as nx

def simulate_trajectory(P, pi0, size):
    """
    Simula uma trajetória de tamanho "size" onde o estado inicial é escolhido
    baseado em uma distribuição de probabilidade inicial pi0.
    """
    n = P.shape[0]

    current_state = np.random.choice(n, p=pi0)
    trajectory = [current_state]


    for _ in range(size):
        next_state = np.random.choice(n, p=P[current_state])
        trajectory.append(next_state)
        current_state = next_state
        
    return trajectory

def diagnose_with_graphs(P):
    # Grafo
    G = nx.from_numpy_array(P, create_using=nx.DiGraph)
        
    # Irredutibilidade (Fortemente Conexo)
    is_irreducible = nx.is_strongly_connected(G)
    print(f"Irredutível: {is_irreducible}")
    
    # Aperiodicidade
    # Um grafo é aperiódico se o MDC dos comprimentos de todos os ciclos é 1
    if is_irreducible:
        is_aperiodic = nx.is_aperiodic(G)
        print(f"Aperiódica: {is_aperiodic}")
    else:
        # Se não for irredutível, a aperiodicidade deve ser checada por componente
        is_aperiodic = all(nx.is_aperiodic(G.subgraph(c)) 
                          for c in nx.strongly_connected_components(G))
        print(f"Aperiódica: {is_aperiodic}")

    # Análise de Componentes
    scc = list(nx.strongly_connected_components(G))
    print(f"Número de classes comunicantes: {len(scc)}")


if __name__ == "__main__":
    n= 4
    size= 10000

    # Simulação de uma trajetória
    pi0 = generate_probability_vector(n, 1)
    P = generate_transition_matrix(n, 1)
    diagnose_with_graphs(P)

    # # Matriz redutível
    # P= np.array([
    #     [0.8, 0.2, 0.0, 0.0],
    #     [0.1, 0.9, 0.0, 0.0],
    #     [0.0, 0.0, 0.7, 0.3],
    #     [0.0, 0.0, 0.4, 0.6]
    # ])

    # # Cadeia periódica
    # P= np.array([
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1],
    #     [1, 0, 0, 0]
    # ])
    mark1 = simulate_trajectory(P, pi0, size)

    counts = np.bincount(mark1, minlength=n)
    empirical_dist = counts / size

    # Solução teórica
    A, b= build_steady_state_system(P)
    Q, R= qr_decomposition(A)
    sol = solver_qr(Q,R,b)

    # comparação
    print(sol)
    print(empirical_dist)

    print(sol @ P) 
    print(empirical_dist @ P)
