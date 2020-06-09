# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:23:37 2019

@author: Arenhart
"""

from truncated_gaussian import *
from laplacian_solver_3 import *
import numpy as np
import matplotlib.pyplot as plt

LAM = 10.68574215283542
SIZE = (10, 10, 10)
PARAMS = (
		('truncated solid', 0.05, (0.01, 0.02, 0.03, 0.04, 0.0499)),
		('truncated solid', 0.15, (0.03, 0.06, 0.09, 0.12, 0.1499)),
		('truncated solid', 0.4, (0.08, 0.16, 0.24, 0.32, 0.399)),
		('compound maps', 0.05, (0.15, 0.30, 0.45, 0.60, 0.75)),
		('compound maps', 0.15, (0.15, 0.30, 0.45, 0.60, 0.75)),
		('compound maps', 0.4, (0.15, 0.30, 0.45, 0.60, 0.75)),
		('added cubes', 0.05, (0.01, 0.02, 0.03, 0.04)),
		('added cubes', 0.15, (0.03, 0.06, 0.09, 0.12)),
		('added cubes', 0.4, (0.08, 0.16, 0.24, 0.32))
		)

COLORS = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')

'''
TGF = TruncatedGaussian(size = SIZE, lam = LAM, porosity = 0.01)
TGF.generate_gaussian_field()
TGF.truncate_field()
field = TGF.field
truncated_field = TGF.truncated_field
rock = SimulatedRock(mode = 'variable', param = TGF.truncated_field, percolation_check = False)
rock.resolve_clay(volumetric_conductivity = 1, surface_conductivity = 10,pore_conductivity = 0.1)
solver = ConductivitySolver(rock)
'''

def test():
    
    VOL_C = 1
    SUR_C = 800
    POR_C = 250
    
    for param in PARAMS*2:
		
        mode, porosity, clay = param
        print(f'{mode}, porosity = {porosity}:')
        TGF = TruncatedGaussian(size = SIZE, lam = LAM,
                                porosity = porosity)
        TGF.generate_gaussian_field()
        TGF.truncate_field()
        clean_field = TGF.truncated_field.copy()
        rock = SimulatedRock(mode = 'variable', param = clean_field, 
                             percolation_check = False)
        rock.resolve_clay(volumetric_conductivity = VOL_C, 
                          surface_conductivity = SUR_C,
                          pore_conductivity = POR_C)
        solver = ConductivitySolver(rock)
        print(f'{mode}, porosity = {porosity}, 0 clay:'+str(solver.solve_laplacian()))
        
        for c in clay:
            TGF.truncated_field = clean_field.copy()
            TGF.add_clay( fraction = c, mode = mode)
            rock = SimulatedRock(mode = 'variable', 
                                 param = TGF.truncated_field,
                                 percolation_check = False)
            rock.resolve_clay(volumetric_conductivity = VOL_C, 
                              surface_conductivity = SUR_C,
                              pore_conductivity = POR_C)
            solver = ConductivitySolver(rock)
            print(f'{mode}, porosity = {porosity}, {c} clay:' + str(solver.solve_laplacian()))
    input('Press enter to exit')

def load_results():
	

	results = {'truncated solid' : [],
			   'compound maps': [],
			   'added cubes': []}
	
	with open('results_cilamce.txt', mode = 'r') as file:
		
		for line in file:
			columns = line.split(',')
			mode = columns[0]
			porosity = float(columns[1].split('=')[1].strip())
			clay = float(columns[2].strip().split(' ')[0].strip())
			conductivity = float(columns[3].strip()[:-1])
			if conductivity < 0:
				conductivity = 0
			log_conductivity = np.log10(conductivity)
			normalized_clay = clay/porosity
			results[mode].append([porosity, clay, conductivity, 
		                          normalized_clay, log_conductivity])
			
	return results

def analyze_results(results):
	
	output = {}
	
	for mode in results:
		
		output[mode] = {}
		
		all_data = np.array(results[mode])
		for porosity in np.unique(all_data[:,0]):
			rows = all_data[all_data[:,0] == porosity]
			x = rows[:,1]
			x /= x.max()
			y = rows[:,2]/100
			x_mean = np.unique(x)
			y_mean = np.zeros(len(x_mean))
			for i in range(len(x_mean)):
				mean = y[x==x_mean[i]].mean()
				y_mean[i] = mean
			output[mode][porosity] = (x, y, x_mean, y_mean)
			
			
	for mode in output:
		for porosity, color in zip(output[mode], COLORS):
			x, y, x_mean, y_mean = output[mode][porosity]
			plt.plot(x, y, f'o{color}', x_mean, y_mean, f'-{color}')
		plt.title(mode)
		plt.yscale('log')
		plt.ylabel('Formation Factor [log10(S/e-6 m)]')
		plt.xlabel('Relative clay content')
		plt.savefig(f'{mode}.png')
		plt.show()
		
	return output
	
			
	
	
	
	
	