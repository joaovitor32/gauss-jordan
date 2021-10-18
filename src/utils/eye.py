import numpy as np


def eye(rows_qtd: int) -> np.array:
    eye_array = []
    for number in range(rows_qtd):
        eye_array.append(np.eye(1, rows_qtd, number)[0])

    return np.array(eye_array)
