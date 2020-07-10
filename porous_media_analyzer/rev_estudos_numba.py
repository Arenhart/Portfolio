# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:10:31 2020

@author: Rafael Arenhart
"""

from skimage import draw
from skimage import filters
from skimage import morphology
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import io
from scipy import ndimage
from numba import jit, njit, prange, int32, int64, uint32
from numba.typed import Dict
import tkinter.filedialog as filedialog
import PIL.Image as pil
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm, inch
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

DICT_PATH = 'dimensions_list.csv'

def add_region(arr, position, size, porosity, shape):
	 
	view = arr[position[0] : position[0] + size[0], 
			   position[1] : position[1] + size[1]]
	
	target_porosity= - math.log(1-porosity)
	target_area = size[0] * size[1] * target_porosity
	pore_area = shape[0] * shape[1] * math.pi
	n_pores = int(target_area // pore_area)
	
	for _ in range(n_pores):
		
		x = np.random.randint(0,size[0])
		y = np.random.randint(0,size[1])
		
		rr, cc = draw.ellipse(x, y, shape[0], shape[1], shape = size)
		view[rr, cc] = 1
		
def old_correlation(arr):
	'''
	Legacy code, only for binary correlation
	'''
	
	x, y = arr.shape
	corr_x = np.zeros(x//2)
	corr_y = np.zeros(y//2)
	corr_x[0] = arr.sum()/arr.size
	corr_y[0] = arr.sum()/arr.size
	
	for i in range(1,x//2):
		corr_x[i] = (arr[i:,:] * arr[:-i,:]).sum() / (arr.size - arr.shape[1] * i)
		
	for i in range(1,y//2):
		corr_y[i] = (arr[:,i:] * arr[:,:-i]).sum() / (arr.size - arr.shape[0] * i)
		
	return corr_x, corr_y

def covariance_3d(arr):
	'''
	3D correlation for scalar arrays
	'''
	
	x, y, z = arr.shape
	corr = np.zeros((3, max(x,y,z)-1))
	corr[:,0] = 1
	
	for ax in (0,1,2):
		for i in range(1, arr.shape[ax] - 1):
			null_index = slice(None, None, None)
			left_index = slice(i, None)
			right_index = slice(-i)
			left_slice = [null_index,] * 3
			left_slice[ax] = left_index
			left_slice = tuple(left_slice)
			right_slice = [null_index,] * 3
			right_slice[ax] = right_index
			right_slice = tuple(right_slice)
			corr[ax][i] =1 - ((((arr[left_slice] - arr[right_slice])**2).sum()  / (arr[left_slice]).size)
			                        / limit_covariance(arr[left_slice],arr[right_slice]))
			
	return corr

def limit_covariance(arr1, arr2):
	low = min(arr1.min(), arr2.min())
	high = max(arr2.max(), arr2.max())
	bins = int(min(high - low + 1, np.sqrt(arr1.size + arr2.size)))
	count_1, edges_1 = np.histogram(arr1, bins = bins)
	count_2, edges_2 = np.histogram(arr2, bins = bins)
	
	if 'int' in str(arr1.dtype):
		disp = edges_1[1]
		edges_1 -= disp
		edges_1+= 2*disp * np.linspace(0, 1, num = edges_1.size)
		
	if 'int' in str(arr2.dtype):
		disp = edges_2[1]
		edges_2 -= disp
		edges_2 += 2*disp * np.linspace(0, 1, num = edges_2.size)
	
	total_1 = count_1.sum()
	count_1 = count_1 / total_1
	total_2 = count_2.sum()
	count_2 = count_2 / total_2
	covariance = 0
	for i, j in ((a,b) for a in range(bins) for b in range(bins)):
		mean_i = (edges_1[i] + edges_1[i+1]) / 2
		mean_j = (edges_2[j] + edges_2[j+1]) / 2
		probability = count_1[i] * count_2[j]
		value = (mean_i - mean_j) ** 2
		covariance += probability * value
	
	return covariance
	
'''
def correlation(arr):
	
	x, y = arr.shape
	corr_x = np.zeros(x)
	corr_y = np.zeros(y)
	
	for i in range(x):
		corr_y += np.correlate(arr[i,:], arr[i,:], mode = 'full')[-y:]
	corr_y /= arr.size
		
	for i in range(y):
		corr_x += np.correlate(arr[:,i], arr[:,i], mode = 'full')[-x:]
	corr_x /= arr.size
		
	return corr_x, corr_y

def vector_correlation(v):
	
	corr = []
	
	corr.append(np.sum(v*v))
	for i in range(1,len(v)):
		
		corr.append(np.sum(v[i:] * v[:-i]))
'''
		
def draw_image(arr, name = 'default.png'):
	
	fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
	corr_x, corr_y = old_correlation(arr)
	phi = arr.mean()
	phi2 = phi**2
	x_size = max((len(corr_x), len(corr_y)))
	
	ax1.imshow(arr)
	ax1.axis('off')
	ax2.plot(corr_x)
	ax2.plot(corr_y)
	ax2.plot((0,x_size), (phi2, phi2), color = '0.75', ls = '--')
	plt.savefig(f'REV/{name}')
	
def rve_porosity(img, individual_results = False):
	
	x, y, z = img.shape
	divisors =  np.array(img.shape)
	divisors = np.log2(divisors).astype('uint16')
	total_divisions = divisors.min()
	extra_divisions = divisors - total_divisions
	results = []
	partial_results = []
	for i in range(1, total_divisions):
		partial = []
		for steps in ((x_step, y_step, z_step) for x_step in range(2**(i+extra_divisions[0]))
		                                                     for y_step in range(2**(i+extra_divisions[1]))
															 for z_step in range(2**(i+extra_divisions[2]))):
			length = 2 ** (total_divisions - i)
			porosity = img[length*steps[0]: length*(1+steps[0]),
				                  length*steps[1]: length*(1+steps[1]),
								  length*steps[2]: length*(1+steps[2])].mean()
			results.append((i, porosity))
			partial.append((porosity))
		partial = np.array(partial)
		partial_results.append((i, partial.mean(), partial.std()))
		
	if individual_results:
		return np.array(partial_results), np.array(results)
	else:
		return np.array(partial_results)

@jit(nopython=True, parallel = False)		
def maxi_balls_serial(img, image_edt):
	out = np.zeros(img.shape)
	distances = image_edt
	for x in prange(img.shape[0]):
		for y in range(img.shape[1]):
			for z in range(img.shape[2]):
				if x == y and y == z and x%10 == 0: print(x)
				radius = distances[x,y,z]
				if radius <= 0.5: continue
				point = np.array((x,y,z)) 
				b_size = int(radius)*2+1
				b = create_sphere(b_size)
				if b.shape[0]%2 == 0: b = b[0:-1,0:-1,0:-1]
				b_radius = b.shape[0] // 2
				lower_bounds = point - b_radius
				upper_bounds = point + b_radius +1
				
				for ax, value in enumerate(lower_bounds):
					if value < 0:
						index = [slice(None, None),] * 3
						index[ax] = slice(-value, None)
						b = b[(index[0],index[1],index[2])]
						lower_bounds[ax] = 0
				for ax, value in enumerate(upper_bounds):
					if value > img.shape[ax]:
						index = [slice(None, None),] * 3
						index[ax] = slice(None, img.shape[ax] - value)
						b = b[(index[0],index[1],index[2])]
						upper_bounds[ax] = img.shape[ax]
				
				
				#image_slice = ([slice(lower_bounds[i], upper_bounds[i]) for i in range(3)])
				#image_slice = (image_slice[0], image_slice[1], image_slice[2])
				#print(type(radius), b.dtype, img.dtype)
				sub_img = img[lower_bounds[0]:upper_bounds[0], lower_bounds[1]: upper_bounds[1], lower_bounds[2]: upper_bounds[2]].astype(np.float64)	
				b_value = b * radius 
				#print(b_value.dtype, sub_img.dtype, b_value.shape, sub_img.shape,'\n')
				b_value = b_value * sub_img
				out[lower_bounds[0]:upper_bounds[0], lower_bounds[1]: upper_bounds[1], lower_bounds[2]: upper_bounds[2]] = np.where(b_value>out[lower_bounds[0]:upper_bounds[0], lower_bounds[1]: upper_bounds[1], lower_bounds[2]: upper_bounds[2]], b_value, out[lower_bounds[0]:upper_bounds[0], lower_bounds[1]: upper_bounds[1], lower_bounds[2]: upper_bounds[2]])
	print('Finished Maxiballs')
	return out


def maxi_balls(img):
	calculated_spheres = Dict.empty(key_type = int64, value_type = uint32[:,:,::1],)
	image_edt = ndimage.morphology.distance_transform_edt(img).astype('uint32')
	return _maxi_balls(img, image_edt, calculated_spheres)


#distances = ndimage.morphology.distance_transform_edt(img)#.astype('uint16')
@jit(nopython=True, parallel = False)		
def _maxi_balls(img, image_edt, calculated_spheres):
	
	x_max, y_max, z_max = img.shape
	out = np.zeros(img.shape).astype(np.uint64)
	distances = image_edt
	progress_steps = (y_max * z_max) // 20
	progress = 0
	for z in prange(z_max):
		for y in range(y_max):
			if ( y +  z  * y_max) % progress_steps == 0:
				progress += 5
				print('Maxi balls conversion:', progress, ' %')
			for x in range(x_max):
				
				radius = distances[x,y,z]
				if radius <= 0.5: continue
				point = np.array((x,y,z)) 
				b_size = int(radius)*2+1
				
				registered_key = 0
				for key in calculated_spheres.keys():
					if b_size == key:
						b = calculated_spheres[b_size]
						registered_key = 1
						break
				if registered_key == 0:
					calculated_spheres[b_size] = create_sphere(b_size)
					b = calculated_spheres[b_size]
				
				if b.shape[0]%2 == 0: b = b[0:-1,0:-1,0:-1]
				b_radius = b.shape[0] // 2
				lower_bounds = point - b_radius
				upper_bounds = point + b_radius +1
				
				for ax, value in enumerate(lower_bounds):
					if value < 0:
						index = [slice(None, None),] * 3
						index[ax] = slice(-value, None)
						b = b[(index[0],index[1],index[2])]
						lower_bounds[ax] = 0
				for ax, value in enumerate(upper_bounds):
					if value > img.shape[ax]:
						index = [slice(None, None),] * 3
						index[ax] = slice(None, img.shape[ax] - value)
						b = b[(index[0],index[1],index[2])]
						upper_bounds[ax] = img.shape[ax]
				
				
				sub_out = out[lower_bounds[0]:upper_bounds[0], lower_bounds[1]: upper_bounds[1], lower_bounds[2]: upper_bounds[2]]
				sub_img = img[lower_bounds[0]:upper_bounds[0], lower_bounds[1]: upper_bounds[1], lower_bounds[2]: upper_bounds[2]]
				b_value = b * radius 
				b_value = b_value * sub_img
				#print(sub_out, b_value, radius, radius <= 0.5, (x,y,z))
				inscribe_spheres(sub_out, b_value, b.shape)

	print('Finished Maxiballs')
	return out

@njit(parallel = False)
def inscribe_spheres(img, values, shape):
	
	for x in prange(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				if values[x,y,z] > img[x,y,z]:
					img[x,y,z] = values[x,y,z]
					

@jit(nopython=True)
def create_sphere(diameter):
	
	ind = np.zeros((3,diameter, diameter, diameter))
	for i in range(1, diameter):
		ind[0,i,:,:] = i
		ind[1,:,i,:] = i
		ind[2,:,:,i] = i
	ind -= (diameter - 1) / 2
	ball = np.sqrt(ind[0,:,:,:]**2 + ind[1,:,:,:]**2 + ind[2,:,:,:]**2)
	radius = (diameter-1)/2
	ball = (ball <= radius).astype(np.uint32)
	return ball

def run(debug = False):
	

	samples = []
	sample_names = filedialog.askopenfilenames()
		
	saved_dimensions = {}
	
	try:
		with open(DICT_PATH, mode = 'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				x, y, z = [int(i.strip()) for i in row[1].split(',')]
				saved_dimensions[row[0]] = (x, y, z)
	except FileNotFoundError:
		pass
	
	modified_dict = False
	for sample in sample_names:
		if sample in saved_dimensions.keys():
			samples.append((sample, saved_dimensions[sample]))
		else:
			dimensions = input(f'Amostra: {sample} Dimensões ( [x,y,z] separadas por virgula): ')
			x, y, z = [int(i.strip()) for i in dimensions.split(',')]
			samples.append((sample, (x, y, z)))
			saved_dimensions[sample] = f'{x},{y},{z}'
			modified_dict = True

	if modified_dict:
		with open(DICT_PATH, mode = 'w', newline = '') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for key in saved_dimensions.keys():
				writer.writerow([key, saved_dimensions[key]])
	
	for sample in samples:
		path = sample[0]
		x, y, z = sample[1]
		img = np.fromfile(path, dtype = 'uint8')
		img = img.reshape((z,y,x), order = 'C')
		img = np.transpose(img)
		
		mb_img = maxi_balls(img)
		results = covariance_3d(mb_img)
		hist = np.histogram(mb_img.flatten(), bins = 20)
		np.savetxt(path[:-4]+'_covariogram.csv', results, delimiter = ';')
		np.savetxt(path[:-4]+'_histogram_count.csv', hist[0], delimiter = ';')
		np.savetxt(path[:-4]+'_histogram_edges.csv', hist[1], delimiter = ';')

def plt_as_variable(fig):
	buffer = io.BytesIO()
	fig.savefig(buffer)
	buffer.seek(0)
	return pil.open(buffer)
	
def consolidate_covariogram_results():
	folder = filedialog.askdirectory()
	reference_file = 'dimensions_list.csv'
	str_cov = '_covariogram.csv'
	str_hist_count = '_histogram_count.csv'
	str_hist_edges = '_histogram_edges.csv'
	
	volumes = {}
	with open(folder + '//' + reference_file, newline = '', encoding = 'utf-8') as file:
		csv_reader = csv.reader(file, delimiter = ',')
		for line in csv_reader:
			key = line[0].split('/')[-1][:-4]
			val = {}
			val['size'] = eval(f'({line[1]})')
			val['res'] = int(line[2])
			for new_key, i in (('rock_type', 3),('pore_type', 4),('pore_size',5)):
				try:
					val[new_key] = line[i]
				except IndexError:
					val[new_key] = 'Indefinido'
			volumes[key] = val
	
	for sample in volumes.keys():
		with open(f'{folder}//{sample}{str_cov}', newline = '') as file:
			csv_reader = csv.reader(file, delimiter = ';')
			volumes[sample]['covariogram_x'] = [float(i) for i in next(csv_reader)]
			volumes[sample]['covariogram_y'] = [float(i) for i in next(csv_reader)]
			volumes[sample]['covariogram_z'] = [float(i) for i in next(csv_reader)]
		with open(f'{folder}//{sample}{str_hist_count}', newline = '') as file:
			count = [float(i) for i in file]
			volumes[sample]['histogram_count'] = count
		with open(f'{folder}//{sample}{str_hist_edges}', newline = '') as file:
			count = [float(i) for i in file]
			volumes[sample]['histogram_edges'] = count
	
	return volumes
	
def color_gen():
	colors = ('blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkgreen', 'gold', 'crimson', 'tomato', 'pink', 'indigo')
	i = 0
	while i < len(colors):
		yield colors[i]
		i += 1
	
def generate_report(volumes, output_file = 'report.pdf'):
	c = canvas.Canvas(output_file)
	
	for sample in volumes.keys():
		c.drawString(3*cm, 27*cm, f'Amostra: {sample}')
		c.drawString(3*cm, 4*cm, f'Tipo de rocha: {volumes[sample]["rock_type"]}')
		c.drawString(3*cm, 3.5*cm, f'Tipo de poro: {volumes[sample]["pore_type"]}')
		c.drawString(3*cm, 3*cm, f'Tamanho de poro: {volumes[sample]["pore_size"]}')
		fig = plt.figure(dpi = 300, figsize = [15 * (cm/inch), 18 * (cm/inch)])
		ax1 = fig.add_subplot(211, title = 'Covariograma', xlabel = 'Distância [nm]', ylabel = 'Covariância')
		ax2 = fig.add_subplot(212, title = 'Distribuição de tamanho de poro', xlabel = 'Tamanho de poro [nm]', ylabel = 'Frequência')
		res = volumes[sample]['res']
		for cov in ('covariogram_x', 'covariogram_y', 'covariogram_z'):
			vals = volumes[sample][cov]
			x_axis = [i*res for i in range(len(vals))]
			ax1.plot(x_axis, vals, label=cov)
		ax1.legend()
		edges = volumes[sample]['histogram_edges']
		bins = [((edges[i] + edges[i+1])/2)*res for i in range(len(edges)-1)]
		count = volumes[sample]['histogram_count']
		for i in range(1, len(count)-1):
			if count[i] == 0 and count[i-1] >0 and count[i+1]>0:
				count[i] = (count[i-1] + count[i+1]) / 2
		bins = bins[1:]
		count = count[1:]
		ax2.plot(bins, count)
		img = ImageReader(plt_as_variable(fig))
		c.drawImage(img, 3*cm, 5*cm, width = cm*15, height = cm*20)
		c.showPage()
		plt.close(fig)
	
	for prop in ('rock_type','pore_type','pore_size'):
		c.setStrokeColor('black')
		c.drawString(3*cm, 27*cm, f'Propriedade: {prop}')
		fig = plt.figure(dpi = 300, figsize = [15 * (cm/inch), 18 * (cm/inch)])
		ax1 = fig.add_subplot(211, xscale = 'log')
		ax2 = fig.add_subplot(212, xscale = 'log')
		col_gen = color_gen()
		vals = [volumes[i][prop] for i in volumes.keys()]
		vals = list(set(vals))
		legend_pos = 4
		
		for val in vals:
			color = next(col_gen)
			c.setStrokeColor(color)
			c.setFillColor(color)
			c.drawString(3*cm, legend_pos*cm, f'{val}')
			legend_pos -= 0.5
			samples = [i for i in volumes.keys() if volumes[i][prop] == val]
			for sample in samples:
				res = volumes[sample]['res']
				vals = volumes[sample]['covariogram_z']
				x_axis = [i*res for i in range(len(vals))]
				ax1.plot(x_axis, vals, color = color, linewidth = 1)
				edges = volumes[sample]['histogram_edges']
				bins = [((edges[i] + edges[i+1])/2)*res for i in range(len(edges)-1)]
				count = volumes[sample]['histogram_count']
				for i in range(1, len(count)-1):
					if count[i] == 0 and count[i-1] >0 and count[i+1]>0:
						count[i] = (count[i-1] + count[i+1]) / 2
				bins = bins[1:]
				count = count[1:]
				total = sum(count)
				count = [i/total for i in count]
				ax2.plot(bins, count, color = color, linewidth = 1)
			
		img = ImageReader(plt_as_variable(fig))
		c.drawImage(img, 3*cm, 5*cm, width = cm*15, height = cm*20)
		c.showPage()
		plt.close(fig)	
		
	c.save()
	
	
'''
Examples:
arr = np.zeros((2000,2000))
add_region(arr, (0,0), (2000,2000), 0.55, (20,10))
draw_image(arr, 'teste3.png')

arr = np.zeros((2000,2000))
add_region(arr, (0,0), (1000,2000), 0.55, (20,10))
add_region(arr, (1000,0), (1000,2000), 0.25, (60,60))
draw_image(arr, 'teste4.png')

arr = np.zeros((2000,2000))
add_region(arr, (0,0), (1000,2000), 0.55, (20,10))
add_region(arr, (1000,0), (1000,2000), 0.25, (5,5))
draw_image(arr, 'teste5.png')

arr = np.zeros((2000,2000))
add_region(arr, (0,0), (1000,2000), 0.55, (20,10))
add_region(arr, (1000,0), (1000,2000), 0.55, (5,5))
draw_image(arr, 'teste6.png')

arr = np.zeros((2000,2000))
add_region(arr, (0,0), (2000,2000), 0.35, (3,3))
add_region(arr, (0,0), (2000,2000), 0.25, (100,100))
draw_image(arr, 'teste1.png')

arr = np.zeros((2000,2000))
add_region(arr, (0,0), (300,2000), 0.25, (10,10))
add_region(arr, (300,0), (300,2000), 0.55, (3,3))
add_region(arr, (600,0), (100,2000), 0.25, (10,10))
add_region(arr, (700,0), (500,2000), 0.55, (3,3))
add_region(arr, (1200,0), (300,2000), 0.25, (10,10))
add_region(arr, (1300,0), (200,2000), 0.55, (3,3))
add_region(arr, (1500,0), (500,2000), 0.25, (10,10))
draw_image(arr, 'teste2.png')

arr = np.zeros((2000,2000))
add_region(arr, (0,0), (2000,2000), 0.25, (10,10))
add_region(arr, (100,100), (400,400), 0.55, (3,3))
add_region(arr, (50,600), (400,400), 0.55, (3,3))
add_region(arr, (1500,600), (400,400), 0.55, (3,3))
add_region(arr, (1000,100), (400,400), 0.55, (3,3))
add_region(arr, (1000,1000), (400,400), 0.55, (3,3))
add_region(arr, (1300,500), (400,400), 0.55, (3,3))
add_region(arr, (500,1300), (400,400), 0.55, (3,3))
draw_image(arr, 'teste7.png')

arr = np.zeros((2000,2000))
arr[300:600,:] = 1
arr[700:1200,:] = 1
arr[1300:1500,:] = 1
draw_image(arr, 'teste8.png')

sample image
path = r'D:\\Desktop\\amostra imagem'
dol = np.fromfile(path+r'\dol_crop.raw', dtype = 'int16')
dol = dol.reshape((512,512,256), order = 'F')
thresh = filters.threshold_otsu(dol)
dol = (dol <= thresh).astype('uint8')

dol_p = maxi_balls(dol)
dol_n = maxi_balls(1 - dol)
dol_np = dol_p - dol_n

cov_r = covariance_3d(dol)
cov_p = covariance_3d(dol_p)
cov_n = covariance_3d(dol_n)
cov_np = covariance_3d(dol_np)
2 * p * (1-p) #dol_r limit covariance
limit_covariance(dol_p)
limit_covariance(dol_n)
limit_covariance(dol_np)
for i,j in (('x',0),('y',1),('z',2)):
    np.savetxt(f'cov_p_{i}.txt', cov_p[j], delimiter = ';')
    
for i,j in (('x',0),('y',1),('z',2)):
    np.savetxt(f'cov_n_{i}.txt', cov_n[j], delimiter = ';')
    
for i,j in (('x',0),('y',1),('z',2)):
    np.savetxt(f'cov_np_{i}.txt', cov_np[j], delimiter = ';')
    
for i,j in (('x',0),('y',1),('z',2)):
    np.savetxt(f'cov_r_{i}.txt', cov_r[j], delimiter = ';')
    
plt.imsave('dol_np_slice.png', dol_np[:,:,100], cmap = 'seismic')
plt.imsave('dol_n_slice.png', dol_n[:,:,100], cmap = 'plasma')
plt.imsave('dol_p_slice.png', dol_p[:,:,100], cmap = 'plasma')

mb = maxi_balls(arr,edt)
maxi_balls.parallel_diagnostics(level=2)
'''





