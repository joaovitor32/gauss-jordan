import numpy as np
from libs.GaussJordanQuadrado import gaussJordanQuadrado
from utils.rank import rank
from utils.inversa import inversa

ResponseType = type({"b": np.array, "R": np.array})


def gaussJordanRetangular(matriz: np.array, b: np.array) -> ResponseType:
    rank_matriz = rank(matriz)

    solved_matriz = gaussJordanQuadrado(matriz, b)

    B = solved_matriz[:, :rank_matriz]
    N = solved_matriz[:, rank_matriz:-1]
    b = solved_matriz[:, -1]

    rows_qtd, cols_qtd = B.shape

    if not np.array_equal(np.identity(cols_qtd), np.dot(inversa(B), B)):
        raise Exception("Condition I=B^(-1)B is not satisfied")

    if not np.array_equal(N, np.dot(inversa(B), N)):
        raise Exception("Condition R=B^(-1)N is not satisfied")

    if not np.array_equal(b, np.dot(inversa(B), b)):
        raise Exception("Condition b=B^(-1)b is not satisfied")

    return {"b": b, "R": N}
