# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:49:24 2019

@author: Rafael Arenhart
"""

import numpy as np
from scipy import ndimage, spatial
import PIL.Image as pil
from skimage import filters, morphology

# Ceragar imagem e converter em uma matriz numpy
img = pil.open('color_balls.png')
arr = np.array(img)

# cria um nva métrica: a "pureza" da cor, igual a diferença entre o maior e o
# menor valor da matriz RGB

max_array = arr.max(axis=2).astype('float32')
min_array = arr.min(axis=2).astype('float32')
mid_array = (arr.sum(axis=2) - max_array - min_array).astype('float32')
purity_array = max_array - min_array
purity_array -= purity_array.min()
purity_array = ( purity_array / purity_array.max()  * 255).astype('uint8')

# Identifica o fundo da imagem aplicando uma binarização por Otsu com um limiar
# modificado, e então usa fechamento para remover o ruido da imagem.
threshold = filters.threshold_otsu(purity_array) * 0.75
background_array = (purity_array >= threshold).astype('int8')
background_array = ndimage.morphology.binary_closing(background_array, 
													 morphology.disk(1))

# Identifica e rotula os discos individuais, usando abertura para preencher as
# linhas entre os segmentos do disco, e então rotulando por 
disks_array = (1-ndimage.morphology.binary_opening(background_array, 
													 morphology.disk(22)))
labeled_disks, n_disks = ndimage.label(disks_array)
obj_slices = ndimage.find_objects(labeled_disks)

# Para cada disco individual, deleciona apenas a metade da esquerda do quadrado
# que o contém e transforma cada pixel de fundo nessa região em branco
for sl in obj_slices[1:]:
	x_start = sl[0].start
	x_stop = sl[0].stop
	y_start = sl[1].start
	y_stop = sl[1].stop
	y_half_stop = y_start + (y_stop-y_start)//2
	sub_slice = np.s_[x_start:x_stop,y_start:y_half_stop,:]
	sub_arr = arr[sub_slice]
	sub_background = background_array[sub_slice[0:2]]
	for i in range(3):
		sub_arr[:,:,i] *= (1 - sub_background).astype('uint8')
		sub_arr[:,:,i] += (sub_background * 255).astype('uint8')

# Salva a imagem final
img_out = pil.fromarray(arr)
img_out.save('color_balls_output.png')