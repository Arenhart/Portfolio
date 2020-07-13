# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:48:44 2020

@author: Rafael Arenhart
"""

import numpy as np

from jit_minkowsky import get_minkowsky_functionals, cube2index

templates = np.zeros(256, dtype = np.uint8)
with open('minkowsky_templates.csv', mode = 'r') as file:
    for line in file:
        index, n, *temps = [int(i.strip()) for i in line.strip().split(',')]
        if n != len(temps):
            print(f'Wrong Minkowsky templates values: {index}')
        for template in temps:
            if templates[template] != 0:
                print('Repeated template value: {template}')
            templates[template] = index

def find_all_isotropes_of_H():
    '''
    This is necessary since the reference paper was missing template 83
    in configuration 10
    (in addition to template 8 for configuration 1)
    '''
    isotropic_template_H = np.zeros((2,2,2), dtype=np.uint8)
    isotropic_template_H[0, 0, 0] = 1
    isotropic_template_H[1, 0, 0] = 1
    isotropic_template_H[1, 1, 0] = 1
    isotropic_template_H[1, 1, 1] = 1
    img = isotropic_template_H
    templates_H = set()
    for i in range(3):
        img = np.rot90(img, axes = (0,1))
        for _ in range(4):
            img = np.rot90(img, axes = (1,2))
            templates_H.add(cube2index(img))
        img = np.rot90(img, axes = (0,2))
        for _ in range(4):
            img = np.rot90(img, axes = (1,2))
            templates_H.add(cube2index(img))
    isotropic_template_H = np.zeros((2,2,2))
    isotropic_template_H[0, 0, 0] = 1
    isotropic_template_H[0, 1, 0] = 1
    isotropic_template_H[1, 1, 0] = 1
    isotropic_template_H[1, 1, 1] = 1
    img = isotropic_template_H
    for i in range(3):
        img = np.rot90(img, axes = (0,1))
        for _ in range(4):
            img = np.rot90(img, axes = (1,2))
            templates_H.add(cube2index(img))
        img = np.rot90(img, axes = (0,2))
        for _ in range(4):
            img = np.rot90(img, axes = (1,2))
            templates_H.add(cube2index(img))


np.random.seed(42)
N = 10

full_img = np.ones((N, N, N), dtype = np.int8)
random_img = np.random.binomial(1, 0.75, (N, N, N))

print(get_minkowsky_functionals(full_img))
print(get_minkowsky_functionals(random_img))
