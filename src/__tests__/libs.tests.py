# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from libs.GaussJordanRetangular import gaussJordanRetangular
from libs.GaussJordanQuadrado import gaussJordanQuadrado
from unittest.mock import Mock
import unittest
import numpy as np
import yaml

class TestSimple(unittest.TestCase):

    def test_libs_gauss_jordan_quadrado(self):
        A = np.array([[2., 2., 4.], [1., 1., 3.], [1., 3., 4.]])
        b = np.array([10, 9, 17])
        response = np.array(
            [[1., 0., 0., -5.], [0., 1., 0., 2.], [0., 0., 1., 4.]])

        response_matriz = gaussJordanQuadrado(A, b)
        self.assertTrue((response == response_matriz).all())
        pass

    def test_libs_gauss_jordan_retangular(self):
        A = np.array([[1., -2., 5., 3.], [2., -4., 5., 6.], [2., -5., 7., 7.]])
        b = np.array([-4., 2., 3.])
        response = {'b': np.array([-4., -5., -2.]),
                    'R': np.array([[1.], [-1.], [-0.]])}

        response_matriz = gaussJordanRetangular(A, b)

        self.assertTrue((response['b'] == response_matriz['b']).all())
        self.assertTrue((response['R'] == response_matriz['R']).all())
        pass


if __name__ == '__main__':
    unittest.main()
