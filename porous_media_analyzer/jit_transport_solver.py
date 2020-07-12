# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:34:34 2020

@author: Rafael Arenhart
"""

import numpy as np
from numba import njit

NULL_RESISTANCE_FACTOR = 1000

@njit
def coord2index(img, coord):
    x, y, z = img.shape
    ii, jj, kk = coord
    return ii + jj*x + kk*x*y


def calculate_transport(img, **kwargs):
    '''
    Solves the conductivity of the given 3d image
    non-negative values are considered the transport property of the element
    a value of -1 indicates an invalid voxel (outside the actual sample)
    '''

    results = []
    for ax in (0,1,2):
        a, b = _create_diagonal_array(img, axis = ax)
        x = _solve_bicg(a, b)
        flow_array = _create_flow_array(img, x)[:,:,:,ax]
        cross_section = list(flow_array.shape)
        cross_section[ax] -= 1
        length = cross_section.pop(ax)
        total_flow = flow_array.sum()/length
        transport_index = total_flow *((length+2)/(cross_section[0] * cross_section[1]))
        results.append(transport_index)

    return results




@njit
def _create_diagonal_array(img, axis):
    '''
    returns 5 vectors indicationg the 4 diagonals of the A matrix and the b array
    of the equivalent linear equation system
    '''
    x, y, z = img.shape
    diag_main = np.zeros(x*y*z, dtype=np.float32)
    diag_x = np.zeros(x*y*z - 1, dtype=np.float32)
    diag_y = np.zeros(x*y*z -  x, dtype=np.float32)
    diag_z = np.zeros(x*y*z - x*y, dtype=np.float32)
    b = np.zeros(x*y*z, dtype=np.float32)
    null_resistance = img.max()* NULL_RESISTANCE_FACTOR

    for i in range(x):
        for j in range(y):
            for k in range(z):
                if img[i, j, k] == -1:
                    img[i, j, k] = null_resistance

    for i in range(x-1):
        for j in range(y):
            for k in range(z):
                cond1 = img[i, j, k]
                cond2 = img[i+1, j, k]
                if cond1 <= 0 or cond2 <= 0 : continue
                eq_cond = 2/ ((1/cond1) + (1/cond2))
                index = coord2index(img, (i, j, k))
                diag_main[index] -= eq_cond
                diag_main[index + 1] -= eq_cond
                diag_x[index] += eq_cond

    for i in range(x):
        for j in range(y-1):
            for k in range(z):
                cond1 = img[i, j, k]
                cond2 = img[i, j+1, k]
                if cond1 <= 0 or cond2 <= 0 : continue
                eq_cond = 2/ ((1/cond1) + (1/cond2))
                index = coord2index(img, (i, j, k))
                diag_main[index] -= eq_cond
                diag_main[index + x] -= eq_cond
                diag_y[index] += eq_cond

    for i in range(x):
        for j in range(y):
            for k in range(z-1):
                cond1 = img[i, j, k]
                cond2 = img[i, j, k+1]
                if cond1 <= 0 or cond2 <= 0 : continue
                eq_cond = 2/ ((1/cond1) + (1/cond2))
                index = coord2index(img, (i, j, k))
                diag_main[index] -= eq_cond
                diag_main[index + x*y] -= eq_cond
                diag_z[index] += eq_cond

    if axis == 0:
        for j in range(y):
            for k in range(z):
                if img[0, j, k] > 0:
                    index = coord2index(img, (0, j, k))
                    diag_main[index] -= img[0, j, k]
                if img[x-1, j, k] > 0:
                    index = coord2index(img, (x-1, j, k))
                    diag_main[index] -= img[x-1, j, k]
                    b[index] = - img[x-1, j, k]

    if axis == 1:
        for i in range(x):
            for k in range(z):
                if img[i, 0, k] > 0:
                    index = coord2index(img,(i, 0, k))
                    diag_main[index] -= img[i, 0, k]
                if img[i, y-1, k] > 0:
                    index = coord2index(img,(i, y-1, k))
                    diag_main[index] -= img[i, y-1, k]
                    b[index] = - img[i, y-1, k]

    if axis == 2:
        for i in range(x):
            for j in range(y):
                if img[i, j, 0] > 0:
                    index = coord2index(img,(i, j, 0))
                    diag_main[index] -= img[i, j, 0]
                if img[i, j, z-1] > 0:
                    index = coord2index(img,(i, j, z-1))
                    diag_main[index] -= img[i, j, z-1]
                    b[index] = - img[i, j, z-1]

    return (diag_main, diag_x, diag_y, diag_z), b

@njit
def _solve_bicg(a_diagonals, b, max_iter = 10**5, tol = 1e-5):
    '''
    solves the linear equation system using the specified format
    '''

    diag_main, diag_x, diag_y, diag_z = a_diagonals

    main_length = diag_main.size
    x_length = diag_x.size
    y_length = diag_y.size
    z_length = diag_z.size

    def sparse_multiplication(vector):

        output = diag_main * vector
        output[:x_length] += diag_x * vector[-x_length:]
        output[-x_length:] += diag_x * vector[:x_length]
        output[:y_length] += diag_y* vector[-y_length:]
        output[-y_length:] += diag_y * vector[:y_length]
        output[:z_length] += diag_z * vector[-z_length:]
        output[-z_length:] += diag_z * vector[:z_length]
        return output

    def get_residual(vector):

        zero_vector = sparse_multiplication(vector)
        zero_vector -= b
        residual = (zero_vector ** 2).mean()
        return residual

    x_0 = np.ones(main_length, dtype= np.float32)
    x_0 /= 2
    x_previous = x_0.copy()

    r_0 = -sparse_multiplication(x_0)
    r_0 += b
    r_previous = r_0.copy()

    rho_previous = alpha = omega_previous = np.float32(1.)
    v_previous = np.zeros(main_length, dtype=np.float32)
    p_previous = np.zeros(main_length, dtype=np.float32)

    for iteraction in range(max_iter):

        rho_next = (r_0 * r_previous).sum()
        beta = (rho_next/rho_previous) * (alpha / omega_previous)
        p_next = r_previous + beta * (p_previous - omega_previous * v_previous)
        v_next = sparse_multiplication(p_next)
        alpha = rho_next / (r_0 * v_next).sum()
        h = x_previous  + alpha*p_next
        residual = get_residual(h)
        if residual <= tol:
            x = h
            break
        s = r_previous - alpha*v_next
        t = sparse_multiplication(s)
        omega_next = (t*s).sum()/(t*t).sum()
        x_next = h + omega_next*s
        residual = get_residual(x_next)
        if residual <= tol:
            x = x_next
            break
        r_next = s - omega_next*t

        #update_index
        r_previous = r_next
        omega_previous = omega_next
        x_previous = x_next
        v_previous = v_next
        rho_previous = rho_next
        p_previous = p_next

    else:
        x = x_next

    return x

@njit
def _create_flow_array(img, potential_vector):
    '''
    takes the solution of linear equations and creates a flow array in shape of
    img
    '''
    x, y, z = img.shape

    potential = np.zeros((x, y, z))
    for i in range(x):
        for j in range(y):
            for k in range(z):
                index = coord2index(potential, (i, j, k))
                potential[i, j, k] = potential_vector[index]

    flow = np.zeros((x, y, z, 3))
    flow[:-1, :, :, 0] = ((potential[1:, :, :] - potential[:-1, :, :])
                              * (2/(1/img[1:, :, :] + 1/img[:-1, :, :])))
    flow[:, :-1, :, 1] = ((potential[:, 1:, :] - potential[:, :-1, :])
                              * (2/(1/img[:, 1:, :] + 1/img[:, :-1, :])))
    flow[:, :, :-1, 2] = ((potential[:, :, 1:] - potential[:, :, :-1])
                              * (2/(1/img[:, :, :1] + 1/img[:, :, :-1])))

    return flow

