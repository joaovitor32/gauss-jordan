# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from utils.eye import eye
from utils.inversa import inversa
from utils.rank import rank
from utils.swap import swap
from unittest.mock import Mock
import unittest
import numpy as np
import yaml

class TestSimple(unittest.TestCase):

    def test_utils_eye(self):
        rows_qtd = 3
        eye_result = eye(rows_qtd)
        self.assertTrue((eye_result == np.identity(rows_qtd)).all())
        pass

    def test_utils_rank(self):
        expected_rank = 2
        matriz = np.array([[1.,0.,1.],[-2.,-3.,1],[3.,3.,-0.]])
        rank_response = rank(matriz)
        self.assertTrue(rank_response==expected_rank)
        pass
    
    def test_utils_inversa(self):
        matriz = np.array([[2,1],[4,3]])
        response = np.array([[3/2,-2],[-1/2,1]])
        
        response_inversa = inversa(matriz)
        
        self.assertTrue((response == response_inversa).all())
        pass
    
    def test_utils_swap(self):
        matriz = np.array([[1,1],[2,2]])
        
        pivot_response =2
        response = np.array([[2,2],[1,1]])
        
        [pivot, matriz_response] = swap(matriz,0)
    
        self.assertTrue(pivot == pivot_response)
        self.assertTrue((response == matriz_response).all())
        pass

if __name__ == '__main__':
    unittest.main()
