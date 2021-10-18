import sys
import yaml
import numpy as np

from libs.GaussJordanQuadrado import gaussJordanQuadrado
from libs.GaussJordanRetangular import gaussJordanRetangular

if __name__ == "__main__":
    yaml_path = './src/data.yaml'

    with open(yaml_path, 'r') as stream:
        try:
            input_data: dict = yaml.safe_load(stream)
            matriz = np.array(input_data['Matriz'])[:, :-1]
            b = np.array(input_data['Matriz'])[:, -1]

        except yaml.YAMLError as error:
            print("[ERROR] Error processing YAML file:", error)
            sys.exit(1)

    rows_qtd, cols_qtd = matriz.shape
    response = gaussJordanQuadrado(
        matriz, b) if rows_qtd == cols_qtd else gaussJordanRetangular(matriz, b)
    print(response)
