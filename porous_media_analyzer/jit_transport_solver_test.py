# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:50:38 2020

@author: Rafael Arenhart
"""

import numpy as np

from jit_transport_solver import _create_diagonal_array, _solve_bicg, _create_flow_array, calculate_transport

np.random.seed(42)
N = 3

full_img = np.ones((N, N, N), dtype = np.float32)
random_img = np.random.random((N, N, N)).astype(np.float32)
random_img *= np.random.binomial(1, 0.8, (N, N, N))

full_a, full_b = _create_diagonal_array(full_img, axis = 1)
full_x = _solve_bicg(full_a, full_b)
full_flow_array = _create_flow_array(full_img, full_x)
full_result = calculate_transport(full_img)

random_a, random_b = _create_diagonal_array(random_img, axis = 1)
random_x = _solve_bicg(random_a, random_b)
random_flow_array = _create_flow_array(random_img, random_x)
random_result = calculate_transport(random_img)