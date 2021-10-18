import numpy as np
from utils.swap import swap


'''
    rank = Rank(matrix)
    The main idea behind this code
    is to apply the echelon form
    and count the number of non-zero
    rows to obtain the rank
'''


def rank(matriz: np.array) -> int:
    rank_number = 0

    copy_matriz = np.matrix.copy(matriz)

    for idx, row in enumerate(copy_matriz):

        a = row[idx]

        if idx == len(matriz)-1:
            break

        if a == 0 and idx < len(matriz)-1:
            [a, copy_matriz] = swap(copy_matriz, idx)

        m = copy_matriz.T[idx]/a

        for idm, _ in enumerate(copy_matriz):
            if idm > idx:
                copy_matriz[idm] -= m[idm]*copy_matriz[idx]

    for row in copy_matriz:
        if np.count_nonzero(row) > 0:
            rank_number += 1

    return rank_number
