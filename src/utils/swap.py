import numpy as np


def swap(matriz: np.array, index: int) -> np.array:
    matriz[[index, index+1]] = matriz[[index+1, index]]
    pivot = matriz[index][index]
    return [pivot, matriz]
