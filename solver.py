import numpy as np

from markov_generators import generate_transition_matrix

def build_steady_state_system(P : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Seja:
    - P, a matriz de transição

    Constrói o sistema linear (I - P)^T * pi = b

    adicionando a restrição sum(pi) = 1.

    Args:
        P (np.ndarray): Matriz de transição (n x n).
        
    Returns:
        tuple: (A, b), 
        
        onde A nxn = (I-P).T com primeira linha substituída por vetor de 1's
             b é nx1 = [1, 0, 0, ..., 0]
    """
    
    n= P.shape[0]

    A= (np.eye(n) - P).T
    b= np.zeros(n)

    A[0, :]= 1.0
    b[0]= 1.0

    return A, b


def qr_decomposition(A : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        Decomposição QR da matriz A (nxn) via reflexões de Householder
    """
    n, m = A.shape
    R = A.copy().astype(float)
    Q = np.eye(n)


    for i in range(n):
        
        x = R[i:, i]
        
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)

        v = x + np.sign(x[0]) * e1

        if np.linalg.norm(v) < 1e-15:
            continue
        
        # Matriz de householder
        H_mini = np.eye(len(v)) - 2.0 * np.outer(v, v) / np.dot(v, v)

        H_k = np.eye(n)
        H_k[i:, i:] = H_mini

        R = H_k @ R
        Q = Q @ H_k

    return Q, R


if __name__ == "__main__":
    P= generate_transition_matrix(dim=3, alpha=3)
    A, b= build_steady_state_system(P)


    print(A)
    print(b)
    Q, R = qr_decomposition(A)
    print(Q)
    print(R)