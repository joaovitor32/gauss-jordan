import numpy as np

from utils.eye import eye


def inversa(matriz: np.array) -> np.array:
    determinante = np.linalg.det(matriz)

    if determinante == 0:
        raise Exception("This matrix does not has inverse")

    rows_qtd, cols_qtd = matriz.shape
    identity_matriz = eye(rows_qtd)
    extended_matriz = np.hstack((matriz.T, identity_matriz))

    for idx, row in enumerate(extended_matriz):
        a = row[idx]
        m = extended_matriz.T[idx]/a

        if a == 0:
            raise ValueError(
                'Pivot number cannot be zero, this matrix does not has inverse')

        for idm, _ in enumerate(extended_matriz):
            if idm != idx:
                extended_matriz[idm] -= m[idm]*extended_matriz[idx]

        extended_matriz[idx] = extended_matriz[idx]/a

    return extended_matriz[:, cols_qtd:]
