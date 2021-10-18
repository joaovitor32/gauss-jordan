import numpy as np

from utils.swap import swap


def gaussJordanQuadrado(matriz: np.array, b: np.array) -> np.array:

    extended_matriz = np.concatenate((matriz, b.reshape((-1, 1))), axis=1)

    for idx, row in enumerate(extended_matriz):
        a = row[idx]
    
        if a == 0 and idx < len(matriz)-1:
            [a, extended_matriz] = swap(extended_matriz, idx)

        m = extended_matriz.T[idx]/a

        for idm, _ in enumerate(extended_matriz):
            if idm != idx:
                extended_matriz[idm] -= m[idm]*extended_matriz[idx]

        extended_matriz[idx] = extended_matriz[idx]/a

    return extended_matriz
