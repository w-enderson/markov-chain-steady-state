import numpy as np

def generate_probability_vector(dim: int, alpha: float =1.0) -> np.ndarray:
    """
    Gera um vetor aleatório ~ Dirichlet(alpha).

    O parâmetro alpha (concentração) define a forma do simplex:
    - alpha > 1: Probabilidades tendem à uniformidade (valores próximos).
    - alpha = 1: Distribuição uniforme no simplex (qualquer combinação é provável).
    - alpha < 1: Probabilidades esparsas (poucos valores altos, muitos próximos de zero).

    Args:
        dim (int): Dimensão do vetor.
        alpha (float): Parâmetro da Dirichlet.

    Returns:
        numpy.ndarray: Vetor de dimensão `dim` cuja soma dos elementos é 1.
    """

    return np.random.dirichlet([alpha] * dim)

def generate_transition_matrix(dim: int, alpha: float=1.0) -> np.ndarray:
    """
    Gera uma matriz de transição (dim x dim).

    Cada linha da matriz é um vetor de probabilidade independente 
    amostrado de uma Dirichlet(alpha).

    Args:
        dim (int): Número de estados (dimensão da matriz quadrada).
        alpha (float): Parâmetro de concentração para cada linha.

    Returns:
        np.ndarray: Matriz de transição de uma Cadeia de Markov.

    """

    matrix = np.array([generate_probability_vector(dim, alpha) for _ in range(dim)])

    return matrix


if __name__ == "__main__":
    # muito esparso (0.001 - 0.1)
    print(generate_transition_matrix(3, 0.1))

    # esparsa (0.1 - 0.9)
    print(generate_transition_matrix(3, 0.7))

    # uniforme (1)
    print(generate_transition_matrix(3, 1))

    # concentrada (1.1 - 10)
    print(generate_transition_matrix(3, 7))
    
    # valores próximos da média (>50)
    print(generate_transition_matrix(3, 60))
