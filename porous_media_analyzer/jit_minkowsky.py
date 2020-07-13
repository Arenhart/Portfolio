# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:48:23 2020

@author: Rafael Arenhart
"""

import numpy as np
from numba import njit, prange

isotropic_templates = np.zeros(256, dtype = np.uint8)
with open('minkowsky_templates.csv', mode = 'r') as file:
    for line in file:
        index, n, *templates = [int(i.strip()) for i in line.strip().split(',')]
        for template in templates:
            isotropic_templates[template] = index

indexing_template = np.zeros((2,2,2), dtype = np.uint8)
for i, j, k in [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]:
    indexing_template[i, j, k] = 2 ** (i + 2*j + 4*k)


with open('minkowsy_values.csv', mode = 'r') as file:
    header = file.readline().strip().split(',')
    minkowsky_names = [i.strip() for i in header[1:]]
    divisors = file.readline().strip().split(',')
    minkowsky_divisors = np.array([int(i) for i in divisors[1:]], dtype=np.int8)
    minkowsky_values = np.zeros((22,6), dtype=np.int64)
    for line in file:
        index, *vals = [int(i.strip()) for i in line.split(',')]
        minkowsky_values[index] = vals

@njit
def cube2index(img):
    return (img * indexing_template).sum()

@njit
def get_minkowsky_functionals(img):

    x, y, z = img.shape

    results = np.zeros(6, dtype=np.int64)
    for i in range(x-1):
        for j in range(y-1):
            for k in range(x-1):
                template_index = cube2index(img[i:i+2,j:j+2,k:k+2])
                minkowsky_template = isotropic_templates[template_index]
                results += minkowsky_values[minkowsky_template]
    results = results / minkowsky_divisors

    return results

@njit(parallel = True)
def get_minkowsky_functionals_parallel(img):

    x, y, z = img.shape

    results = np.zeros(6, dtype=np.int64)
    for i in prange(x-1):
        for j in range(y-1):
            for k in range(x-1):
                template_index = cube2index(img[i:i+2,j:j+2,k:k+2])
                minkowsky_template = isotropic_templates[template_index]
                results += minkowsky_values[minkowsky_template]
    results = results / minkowsky_divisors

    return results
