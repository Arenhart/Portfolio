# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:08:22 2019

@author: Arenhart
"""

import os, math
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats as stats
import matplotlib.pyplot as plt

INVALID = -100
SIGMA = 1.0

def generate_kernel(sigma = 1.0):
	
	sigma_ceil = math.ceil(sigma)
	kernel = np.zeros((sigma_ceil*8+1,)*2)
	kernel[sigma_ceil*4,sigma_ceil*4] = 1
	kernel = ndimage.gaussian_filter(kernel, sigma = sigma)
	kernel /= np.max(kernel)
	
	return kernel


def generate_normalized_maps(filepath_1, filepath_2 = None, step = 1):
	'''
	Returns gap and penetration displacement maps
	Normalizes both input maps and inverts the second
	'''
	
	if filepath_2 == None:
		height_map = read_height_map(filepath_1)
		height_maps = [height_map, height_map.copy()]
	else:
		height_maps = [read_height_map(filepath_1), 
				       read_height_map(filepath_2)]
	
	min_x = min([i.shape[0] for i in height_maps])
	min_y = min([i.shape[1] for i in height_maps])
	
	for i in range(2):
		diff_x = height_maps[i].shape[0] - min_x
		diff_y = height_maps[i].shape[1] - min_y
		if diff_x > 0:
			height_maps[i] = height_maps[i][diff_x//2:(-diff_x)//2,:]
		if diff_y > 0:
			height_maps[i] = height_maps[i][:,diff_y//2:(-diff_y)//2]
			
		height_maps[i] = height_maps[i] - height_maps[i].mean()

	normalized_map = height_maps[0]
	inverted_map = - height_maps[1]
	
	half_x = height_map.shape[0]//2
	half_y = height_map.shape[1]//2
	
	gap_in_x = np.zeros(half_x)
	gap_in_y = np.zeros(half_y)
	
	overlap_in_x = np.zeros(half_x)
	overlap_in_y = np.zeros(half_y)
	
	for i in range(half_x):
		diff = normalized_map[:half_x,:] - inverted_map[i:i+half_x,:]
		sign = diff >= 0
		overlap_in_x[i] = (diff * sign).sum()
		gap_in_x[i] = (diff * (1 - sign)).sum()
		
	for j in range(half_y):
		diff = normalized_map[:,:half_y] - inverted_map[:,j:j+half_y]
		sign = diff >= 0
		overlap_in_y[j] = (diff * sign).sum()
		gap_in_y[j] = (diff * (1 - sign)).sum()	
	
	gap_in_xy = np.zeros((half_x//step,half_y//step))
	overlap_in_xy = np.zeros((half_x//step,half_y//step))
	
	for i,j in ((i,j) for i in range(0,half_x,step) for j in range(0,half_y,step)):
		diff = normalized_map[:half_x,:half_y] - inverted_map[i:i+half_x,j:j+half_y]
		sign = diff >= 0
		overlap_in_xy[i//step,j//step] = (diff * sign).sum()
		gap_in_xy[i//step,j//step] = (diff * (1 - sign)).sum()
		
	return gap_in_xy, overlap_in_xy

			
def read_height_map(filepath):
	
	lines = []
	with open(filepath, mode = 'r') as file:
		for line in file:
			lines.append(line)
	
	number_of_lines = len(lines)
	
	table = np.zeros((number_of_lines,3))
	
	for i,l in enumerate(lines[:-1]):
		l = l.replace('***',str(INVALID))
		table[i,:] = np.array(l.split('\t')).astype('float64')
		
	table[:,0] /= table[1,0]
	table[:,0] = np.rint(table[:,0])
	
	for i in table[:,1]:
		if i != 0:
			table[:,1] /= i
			table[:,1] = np.rint(table[:,1])
			break
	
	Nx = int(np.max(table[:,0]))+1
	Ny = int(np.max(table[:,1]))+1
	height_map = np.zeros((Nx,Ny))
		
	for x, y, v in table:
		height_map[int(x), int(y)] = v
	
	
	kernel = generate_kernel(SIGMA)
	kernel_radius = 4 * math.ceil(SIGMA)
	it = np.nditer(height_map, flags=['multi_index'])
	
	
	while not it.finished:
		
		if it[0] == INVALID:
			x, y = it.multi_index
			x_left = max(x - kernel_radius, 0)
			x_right = min(x + kernel_radius + 1, Nx)
			y_left = max(y - kernel_radius, 0)
			y_right = min(y + kernel_radius + 1, Ny)
			
			sub_map = height_map[x_left: x_right, y_left: y_right]
			sub_kernel = kernel[x_left - x + kernel_radius: x_right - x  + kernel_radius,
						        y_left - y + kernel_radius: y_right - y + kernel_radius]
			
			height_map[x,y] = (np.sum(sub_map * sub_kernel * (sub_map != INVALID)) 
			                   / np.sum(sub_kernel * (sub_map != INVALID)) )
			
		it.iternext()
	
	
	y_slope, y_intercept = stats.linregress(range(height_map.shape[1]),
											height_map.sum(axis = 0))[0:2]
	
	
	
	x_slope, x_intercept = stats.linregress(range(height_map.shape[0]),
										    height_map.sum(axis = 1))[0:2]
	
	
	height_map -= (y_intercept/height_map.shape[0] 
	             + x_intercept/height_map.shape[0])
	
	height_map -= (np.mgrid[0:Nx,:Ny][0] * (x_slope/height_map.shape[1])
			      + np.mgrid[0:Nx,:Ny][1] * (y_slope/height_map.shape[0]))
	
	return height_map

	

	
	
gap_map, overlap_map = generate_normalized_maps(
		                                'Interferometria\\3-CP1-L1-B-SE-2.txt')	
	
	
	