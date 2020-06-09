# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:06:40 2019

@author: Arenhart
"""

from numbers import Number
import numpy as np
import scipy.sparse as sc_sparse
import scipy.sparse.linalg as sc_linalg
import scipy.ndimage as sc_ndimage
import time, random

DEBUG_DURATION = False

class ConductivitySolver():
	
	
	def __init__(self, rock):
		'''
		Solves a numpy array given as input (rock), where each voxel is the 
		elements conductivity
		'''
		self.removed_sites = []
		self.dir_dict = {'+x' : 0,
						 '+y' : 1,
						 '+z' : 2,
						 '-x' : 3,
						 '-y' : 4,
						 '-z' : 5}
		
		self.inverse_dict = {'+x' : '-x',
							 '+y' : '-y',
							 '+z' : '-z',
							 '-x' : '+x',
							 '-y' : '+y',
							 '-z' : '+z'}
		
		self.displacement_dict = {
							 '-x': (-1,0,0),
							 '-y': (0,-1,0),
							 '-z': (0,0,-1),
							 '+x': (1,0,0),
							 '+y': (0,1,0),
							 '+z': (0,0,1)}
		
		self.rock = rock

   
	def get_directional_cond(self, cond ,direction):
		
		if isinstance(cond, Number): return cond
		
		return cond[self.dir_dict[direction]]
	
	def make_empty_sparse(self):
		'''
		creates a csr empty matrix optimized for 6-neghbourhood porous systems
		'''
		x, y, z = self.rock.shape
		size = x * y * z
		diagonals = [-x*y, -x, -1, 0, 1, x, x*y]
		elements = sum([(size-abs(d)) for d in diagonals])
		dat = np.zeros(elements,dtype='float32')
		ind = np.zeros(elements,dtype='int32')
		indptr = np.zeros((size+1),dtype='int32')
		
		i=0
		for row in range(size):
			for disp in diagonals:
				if (row + disp) >= 0 and (row + disp) < size:
					ind[i] = (row + disp)
					i += 1
			indptr[row+1] = i
			
		return sc_sparse.csr_matrix((dat,ind,indptr),shape = (size,size))

	
	def make_operator_array(self, sparse = 'csr predefined'):
		
		if sparse == 'csr predefined':
			self.operator_array = self.make_empty_sparse()
		else:
			raise Exception
			
		N = self.rock.size
		x,y,z = self.rock.shape
		array = self.rock
		
		diag_main = np.zeros(N) # main diagonal
		
		diag_x = np.array(
						1/ 
				(0.5/np.concatenate((array[1:,:,:],np.zeros((1,y,z))),
									axis=0)
					+
				(0.5/array ))).flatten(order = 'F')
		diag_main[:-1] -= diag_x[:-1]
		diag_main[1:] -= diag_x[:-1]
		
		self.operator_array.setdiag(diag_x[:-1],k=1)
		self.operator_array.setdiag(diag_x[:-1],k=-1)
		
		diag_y = np.array(
						1/ 
				(0.5/np.concatenate((array[:,1:,:],np.zeros((x,1,z))),
									axis=1)
					+
				(0.5/array ))).flatten(order = 'F')
		
		diag_main[:-x] -= diag_y[:-x]
		diag_main[x:] -= diag_y[:-x]
		
		self.operator_array.setdiag(diag_y[:-x],k=x)
		self.operator_array.setdiag(diag_y[:-x],k=-x)
		
		diag_z = np.array(
						1/ 
				(0.5/np.concatenate((array[:,:,1:],np.zeros((x,y,1))),
									axis=2)
					+
				(0.5/array ))).flatten(order = 'F')
		
		diag_main[:-x*y] -= diag_z[:-x*y]
		diag_main[x*y:] -= diag_z[:-x*y]
		
		
		self.operator_array.setdiag(diag_z[:-x*y],k=x*y)
		self.operator_array.setdiag(diag_z[:-x*y],k=-x*y)
		
		diag_main[:x*y] -= array.flatten(order='F')[:x*y]
		diag_main[-x*y:] -= array.flatten(order='F')[-x*y:]
		diag_main += diag_main==0
		
		self.operator_array.setdiag(diag_main,k=0)


	def clean_operator_array(self):
		'''
		Removes insulating elements from array,
		currently not working
		'''
		
		if self.removed_sites != []:
			print ('operator array already cleaned')
			return
		for i in range(self.operator_array.shape[0]-1,-1,-1):
			if self.operator_array[i,i] == 1:
				self.operator_array = np.delete(self.operator_array,i,0)
				self.operator_array = np.delete(self.operator_array,i,1)
				self.b_array = np.delete(self.b_array,i)
				self.removed_sites.insert(0,i)
	   
	
	def solve_laplacian(self, clean_array = False, sparse= 'csr predefined',
						estimator_array = 'no', method = 'bcg',
						tol = 1e-07, maxiter = None):
		''' available methods: bcg, bcgstab, cg, minres '''
		
		print('start')
		last_time = time.process_time()
		N = self.rock.size

		x,y,z = self.rock.shape

		self.make_operator_array(sparse)
		
		print('1: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		self.b_array = np.zeros((N,1),dtype='float32')

		if estimator_array == 'yes':
			x0 = self.generate_estimator()
		elif estimator_array == 'simple':
			x0 = np.ones((N,1),dtype='float32')/2
		elif estimator_array == 'no':
			x0 = np.zeros((N,1),dtype='float32')

		print('2: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		for i in range(x*y):
			index = np.unravel_index(i, self.rock.shape)
			if self.get_directional_cond(self.rock[index],'-z') != 0:
				self.b_array[i,0]= -int(self.get_directional_cond(
													self.rock[index],'-z'))
		
		print('3: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		if clean_array:
			start_time = time.clock()
			self.clean_operator_array()
			print('cleaning time', time.clock() - start_time)

		
		print('4: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		if sparse == None:
			pressure_field = np.linalg.solve(self.operator_array,self.b_array)
		else:
			if self.operator_array.getformat() != 'csr':
				self.operator_array = sc_sparse.csr_matrix(self.operator_array)
				
			if method == None:
				pressure_field = sc_linalg.spsolve(self.operator_array,
											   self.b_array)
			elif method == 'bcg':
				pressure_field = sc_linalg.bicg(self.operator_array,
												self.b_array, x0=x0,
												tol = tol, maxiter=maxiter)[0]
			elif method == 'bcgstab':
				pressure_field = sc_linalg.bicgstab(self.operator_array,
												self.b_array, x0=x0,
												tol = tol, maxiter=maxiter)[0]
			elif method == 'cg':
				pressure_field = sc_linalg.cg(self.operator_array,
												self.b_array, x0=x0,
												tol = tol, maxiter=maxiter)[0]
			elif method == 'minres':
				pressure_field = sc_linalg.minres(self.operator_array,
												self.b_array, x0=x0,
												tol = tol, maxiter=maxiter)[0]
				
			else:
				raise Exception

		
		print('5: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		for i in self.removed_sites:
			pressure_field = np.insert(pressure_field,i,0)
   
		pressure_field_shaped = np.zeros((x,y,z))

		for i in range(N):
			xyz = (i%x, (i//x)%y, i//(x*y))
			pressure_field_shaped[xyz] = pressure_field[i]
		
		print('6: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		velocity_field = np.zeros((x,y,z))

		for index in ((a,b,c) for a in range(x)
							  for b in range(y)
							  for c in range(z)):
			if index[2] == 0:
				cond1 = self.rock[index]
				c = self.get_directional_cond(cond1,'-z')
				velocity_field[index] = (1 - pressure_field_shaped[index]) * c
			elif index[2] == (z-1):
				continue
			else:
			   index2 = (index[0],index[1],index[2]+1)
			   cond1 = self.rock[index]
			   cond2 = self.rock[index2]
			   c1 = self.get_directional_cond(cond1,'+z')
			   c2 = self.get_directional_cond(cond2,'-z')
			   if c1 <=0 or c2 <= 0: continue
			   c = 1 / (1/(2*c1) + 1/(2*c2))
			   velocity_field[index] = (pressure_field_shaped[index] - 
							  pressure_field_shaped[index2]) * c 
		
		print('7: ', time.process_time()-last_time)
		last_time = time.process_time()

		avg_velocity = [sum(sum(velocity_field[:,:,i]))/(x*y) 
						for i in range(z)]

		self.velocity_field = velocity_field

		self.pressure_field = pressure_field_shaped

		self.pressure_field_unformated = pressure_field

		#return ('fluxo medio: ', np.mean(avg_velocity[:-1]))
		self.solution = ('Condutividade Laplace: ', 
				np.mean(avg_velocity[:-1])*(z+1))
		self.solution_raw = np.mean(avg_velocity[:-1])*(z+1)
		
		print('8: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		return self.solution
	
	def new_solve_laplacian(self, tol = 1e-07, maxiter = None,
						     clean = False, estimate = False):
		
		
		print('start')
		last_time = time.process_time()

		#TODO make coeficients_array and operator_matrrix local variables after
		#debugging to free RAM
		
		self.coefficients_array = self._make_coefficients_array()
		self.operator_matrix, self.non_empty_elements = self._make_operator_matrix(
				                                                    self.coefficients_array,
																	clean)
		self.b_array = self._make_b_array()
		
		N = self.rock.size
		x,y,z = self.rock.shape
		
		print('1: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		self.x0 = np.zeros((N,1),dtype='float64')
		self.x0 = np.extract(self.non_empty_elements, self.x0)

		print('2: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		if self.rock.size > 1000 and estimate:
			downscale_array = np.zeros((x//2,y//2,z//2))
			percolating_rock = self.get_percolating_array()
			even_percolating_rock = percolating_rock[:2*(x//2),:2*(y//2),:2*(z//2)]
			downscale_array += (even_percolating_rock[::2,::2,::2]
			                  + even_percolating_rock[1::2,::2,::2]
							  + even_percolating_rock[::2,1::2,::2]
							  + even_percolating_rock[1::2,1::2,::2]
							  + even_percolating_rock[::2,::2,1::2]
							  + even_percolating_rock[1::2,::2,1::2]
							  + even_percolating_rock[::2,1::2,1::2]
							  + even_percolating_rock[1::2,1::2,1::2])/8
			estimator = ConductivitySolver(downscale_array)
			estimator.new_solve_laplacian()
			estimated_pressure = estimator.pressure_field
			for i in range(3):
				estimated_pressure = np.repeat(estimated_pressure,2,axis = i)
			est_x, est_y, est_z = estimated_pressure.shape
			if est_x != x:
				estimated_pressure = np.concatenate((estimated_pressure, np.zeros((1,est_y,est_z))), axis = 0)
				estimated_pressure[-1,:,:] = estimated_pressure[-2,:,:]
			if est_y != y:
				estimated_pressure = np.concatenate((estimated_pressure, np.zeros((x,1,est_z))), axis = 1)
				estimated_pressure[:,-1,:] = estimated_pressure[:,-2,:]
			if est_z != z:
				estimated_pressure = np.concatenate((estimated_pressure, np.zeros((x,y,1))), axis = 2)
				estimated_pressure *= (est_z+2)/(est_z+1)
				estimated_pressure[:,:,-1] = estimated_pressure[:,:,-2]/2
			estimated_pressure = estimated_pressure * (percolating_rock != 0)
			self.x0 = estimated_pressure.flatten(order = 'F')
		else:
			self.x0 += 0.5
			'''
			percolating_rock = self.get_percolating_array()
			z_index = np.indices(self.rock.shape)[2]
			x0 = np.ones(self.rock.shape)
			x0 -= (z_index + 1) / (z+1)
			x0 *= percolating_rock > 0
			self.x0 = x0.flatten(order = 'F')
			'''
		
		print('3: ', time.process_time()-last_time)
		last_time = time.process_time()
		

		print('4: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		
		self.b_array = self.b_array.astype('float64')
		
		pressure_field = sc_linalg.bicg(self.operator_matrix,
												self.b_array)[0]#, x0=self.x0)[0]#,
												#tol = tol, maxiter=maxiter)[0]
		
		
		print('5: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		#full_pressure_field = np.zeros(N)
		#np.place(full_pressure_field, self.non_empty_elements, pressure_field)

		pressure_field_shaped = pressure_field.reshape((x,y,z), order = 'F')
		
		print('6: ', time.process_time()-last_time)
		last_time = time.process_time()
		
		velocity_field = (pressure_field_shaped[:,:,:-1]
		                  - pressure_field_shaped[:,:,1:]) * self.coefficients_array['delta_z'].reshape((x,y,z),order = 'F')[:,:,:-1]

		'''
		for index in ((a,b,c) for a in range(x)
							  for b in range(y)
							  for c in range(z)):
			if index[2] == 0:
				cond1 = self.rock[index]
				c = self.get_directional_cond(cond1,'-z')
				velocity_field[index] = (1 - pressure_field_shaped[index]) * c
			elif index[2] == (z-1):
				continue
			else:
			   index2 = (index[0],index[1],index[2]+1)
			   cond1 = self.rock[index]
			   cond2 = self.rock[index2]
			   c1 = self.get_directional_cond(cond1,'+z')
			   c2 = self.get_directional_cond(cond2,'-z')
			   if c1 <=0 or c2 <= 0: continue
			   c = 1 / (1/(2*c1) + 1/(2*c2))
			   velocity_field[index] = (pressure_field_shaped[index] - 
							  pressure_field_shaped[index2]) * c 
	   '''
		
		print('7: ', time.process_time()-last_time)
		last_time = time.process_time()

		avg_velocity = velocity_field.sum(axis=0).sum(axis=0)/(x*y)

		self.velocity_field = velocity_field

		self.pressure_field = pressure_field_shaped

		self.pressure_field_unformated = pressure_field

		self.solution = ('Condutividade Laplace: ', 
				np.mean(avg_velocity[:-1])*(z+1))
		print(self.solution)
		print('8: ', time.process_time()-last_time)
		last_time = time.process_time()
		
	
	def _make_coefficients_array(self):
		
		coef_array_shape = list(self.rock.shape) + [4]
		coef_array = np.zeros(coef_array_shape, dtype = 'float32')
		x, y, z = self.rock.shape
		percolating_rock = self.get_percolating_array()
		
		full_slice = slice(0, None)
		left_slice = slice(0,-1)
		right_slice = slice(1,None)
		
		for displaced_slice, center_slice, i in (
				((right_slice, full_slice, full_slice), (left_slice, full_slice, full_slice), 0),
				((full_slice, right_slice, full_slice), (full_slice, left_slice, full_slice), 1),
				((full_slice, full_slice, right_slice), (full_slice, full_slice, left_slice), 2),
				):
			
			coef_array[center_slice][:,:,:,i] += ( 2 / (1/percolating_rock[center_slice] +
				                                     1/percolating_rock[displaced_slice]))
			
		coef_array[:,:,:,3] -= np.sum(coef_array[:,:,:,0:3],axis = 3)
		
		for right_section, left_section, index in (
				((right_slice,full_slice,full_slice), (left_slice, full_slice, full_slice), 0),
				((full_slice,right_slice,full_slice), (full_slice, left_slice, full_slice), 1),
				((full_slice,full_slice,right_slice), (full_slice, full_slice, left_slice), 2)):
			coef_array[right_section][:,:,:,3] -= coef_array[left_section][:,:,:,index]
				   
		   
		coef_array[:,:,0,3] -= percolating_rock[:,:,0]
		coef_array[:,:,-1,3] -= percolating_rock[:,:,-1]
		
		coefficients = {'delta_x': coef_array[:,:,:,0].flatten(order = 'F'),
				        'delta_y': coef_array[:,:,:,1].flatten(order = 'F'),
					    'delta_z': coef_array[:,:,:,2].flatten(order = 'F'),
					    'center': coef_array[:,:,:,3].flatten(order = 'F')}
		
		return coefficients
	
	def _make_operator_matrix(self, coefficients, clean = False):
		
		if clean:
			non_isolated_elements = coefficients['center'] != 0
		else:
			non_isolated_elements = np.ones(coefficients['center'].shape,
								            dtype = 'uint')
		N = non_isolated_elements.sum()
		x, y, z = self.rock.shape
		
		diagonals = np.zeros((7,N))
		
		diagonals[0,:] = np.extract(non_isolated_elements, coefficients['delta_z'])
		diagonals[1,:] = np.extract(non_isolated_elements, coefficients['delta_y'])
		diagonals[2,:] = np.extract(non_isolated_elements, coefficients['delta_x'])
		diagonals[3,:] = np.extract(non_isolated_elements, coefficients['center'])
		diagonals[4,1:] = np.extract(non_isolated_elements, coefficients['delta_x'])[:-1]
		diagonals[5,x:] = np.extract(non_isolated_elements, coefficients['delta_y'])[:-(x)]
		diagonals[6,x*y:] = np.extract(non_isolated_elements, coefficients['delta_z'])[:-x*y]
		diagonals[3,:][diagonals[3,:] == 0] = 1
		operator_matrix = sc_sparse.dia_matrix((diagonals,(-x*y, -x, -1, 0, 1, x, x*y)), (N,N))
		
		return operator_matrix, non_isolated_elements
	
	def _make_b_array(self):
		
		percolating_rock = self.get_percolating_array()
		x,y,z = self.rock.shape
		
		b_array = np.zeros((x,y,z),dtype='float32')
		b_array[:,:,0] = -percolating_rock[:,:,0]
		b_array = b_array.flatten(order = 'F')
		b_array = np.extract(self.non_empty_elements, b_array)
		
		return b_array
	
	def get_percolating_array(self):
		labeled = sc_ndimage.label(self.rock)[0]
		bottom_labels = np.unique(labeled[:,:,0])
		top_labels = np.unique(labeled[:,:,-1])
		
		percolating_labels = np.intersect1d(
				               bottom_labels, top_labels, assume_unique = True)
		
		return (self.rock *
		        np.isin(labeled, percolating_labels))
	
	def get_cond(self):
		print('VGP not implemented')   
	
	def solve_vgp(self, threshold = 10**-6):
		
		rock_shape = self.rock.shape
		cond_array = np.zeros(rock_shape+(3,),np.float32)
		
		
		for z,y,x in [(i,j,k) for i in range(rock_shape[2])
						for j in range(rock_shape[1]) 
						for k in range(rock_shape[0]) ]:
			
			try:
				dir_cond = self.get_cond((x,y,z), '+x')
				if dir_cond == None: dir_cond = 0 
				cond_array[x,y,z,0] = dir_cond
			except:
				pass
			
			try:
				dir_cond = self.get_cond((x,y,z), '+y')
				if dir_cond == None: dir_cond = 0 
				cond_array[x,y,z,1] = dir_cond
			except:
				pass
			
			try:
				dir_cond = self.get_cond((x,y,z), '+z')
				if dir_cond == None: dir_cond = 0 
				cond_array[x,y,z,2] = dir_cond
			except:
				pass
		
		cond_correction = cond_array.max()*10
		cond_array /= cond_correction
			
		potential_array = np.zeros((rock_shape),np.float32)
		
		for z,y,x in [(i,j,k) for i in range(potential_array.shape[2]) 
							  for j in range(potential_array.shape[1]) 
							  for k in range(potential_array.shape[0]) ]:
			
			potential_array[x,y,z] = 100/potential_array.shape[2] * z
			#initialize potential array with approximate values
		
		last_conductivity = 0

		while True:
			in_curr = 0
			out_curr = 0
			next_array = np.copy(potential_array)
			for z,y,x in [(i,j,k) for i in range(potential_array.shape[2]) 
								  for j in range(potential_array.shape[1]) 
								  for k in range(potential_array.shape[0])]:
				
				
				if z == 0: 
					next_array[x,y,0] = 0
					temp_cond = cond_array[x,y,0,2]
					if temp_cond != 0:
						in_curr += (potential_array[x,y,1]-potential_array[x,y,0]) * temp_cond
					continue
				if z == potential_array.shape[2]-1:
					next_array[x,y,z] = 100
					temp_cond = cond_array[x,y,z-1,2]
					if temp_cond != 0:
						out_curr += (potential_array[x,y,z-1]-potential_array[x,y,z]) * temp_cond
					continue
			
				try:
					temp_cond = cond_array[x,y,z,0]
					next_array[x,y,z] += (potential_array[x+1,y,z]-potential_array[x,y,z]) * temp_cond
				except:
					pass
				try:
					temp_cond = cond_array[x-1,y,z,0]
					next_array[x,y,z] += (potential_array[x-1,y,z]-potential_array[x,y,z]) * temp_cond
				except:
					pass
				try:
					temp_cond = cond_array[x,y,z,1]
					next_array[x,y,z] += (potential_array[x,y+1,z]-potential_array[x,y,z]) * temp_cond
				except:
					pass
				try:
					temp_cond = cond_array[x,y-1,z,1]
					next_array[x,y,z] += (potential_array[x,y-1,z]-potential_array[x,y,z]) * temp_cond
				except:
					pass
				try:
					temp_cond = cond_array[x,y,z,2]
					next_array[x,y,z] += (potential_array[x,y,z+1]-potential_array[x,y,z]) * temp_cond
				except:
					pass
				try:
					temp_cond = cond_array[x,y,z-1,2]
					next_array[x,y,z] += (potential_array[x,y,z-1]-potential_array[x,y,z]) * temp_cond
				except:
					pass
			
			calc_cond = ((in_curr-out_curr)/2) * (cond_correction/100)
			#return (calc_cond, in_curr, out_curr, cond_correction, 
			#		 potential_array, cond_array, next_array)
			potential_array = np.copy(next_array)
			if abs(calc_cond-last_conductivity)/calc_cond <= threshold:
				self.vgp_cond = calc_cond
				return calc_cond / ((rock_shape[0]*rock_shape[1]) /
										 (rock_shape[2]-1))
			last_conductivity = calc_cond
		
	def solve_randomwalk(self,steps=1000, walkers = 100):
		
		x,y,z = self.rock.shape
		squared_distances = []
		
		for _ in range(walkers):
			
			while True:
				
				start_position = (random.randrange(x-1),
							random.randrange(y-1),
							random.randrange(z-1))
				if self.rock[start_position] > 0: break
			position = list(start_position)
			
			for _ in range(steps):
				new_position = list(position)
				new_position[random.randrange(0,3)] += random.randrange(-1,2,2)
				try:
					if self.rock[new_position] > 0:
						position = list(new_position)
				except IndexError:
					pass
				except TypeError:
					pass
					
			distance = sum([(a[0]-a[1])**2 
							for a in zip(start_position,position)])
					
			squared_distances.append(distance)
		
		porosity = (self.rock > 0).sum()/(self.rock > 0).size
		   
		return np.mean(squared_distances) * (porosity/steps)
