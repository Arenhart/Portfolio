# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:11:34 2019

@author: Arenhart
"""

import re, time, sys, math, os, itertools

import numpy as np
from scipy import ndimage, misc, spatial
import matplotlib.pyplot as plt
from skimage import filters, segmentation, morphology, measure, transform
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter as tk
from PIL import Image, ImageTk
from stl import mesh
from numba import jit, njit, prange
from numba.typed import Dict
from tqdm import tqdm

from conductivity_solver import ConductivitySolver
from rev_estudos_numba import maxi_balls

PI = math.pi
MC_TEMPLATES_FILE = 'marching cubes templates.dat'
LOCATION = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

config = {}
with open(os.path.join(LOCATION, 'pma.cfg'), mode='r') as file:
    for line in file:
        key, value = line.split(':')
        key = key.strip()
        value = value.strip()
        config[key] = value

# Helper functions

def face_orientation(v0, v1, v2):
    '''
    Return outward perpendicular vector distance of face along the z axis
    '''
    v0 = np.array(v0),
    v1 = np.array(v1)
    v2 = np.array(v2)
    vector = np.cross(v1 - v0, v2 - v0)
    z_comp = vector[0][2]
    if z_comp > 0.1:
        return -1
    elif z_comp < -0.1:
        return 1
    else:
        return 0


def area_of_triangle(p0, p1, p2):
    '''
    As per Herons formula
    '''
    lines = list(itertools.combinations((p0,p1,p2),2))
    distances = [(spatial.distance.euclidean(i[0],i[1])) for i in lines]
    s = sum(distances)/2
    product_of_diferences = np.prod([(s-i) for i in distances])
    area = math.sqrt(s*product_of_diferences)
    return area


def mc_templates_generator(override = False):
    '''
    Generates a marching cubes template list file, if one is not available
    '''
    if MC_TEMPLATES_FILE in os.listdir(os.getcwd()) and not override:
        return
    summation_to_coordinate = {}
    for i in [(x,y,z) for x in range(2) for y in range(2) for z in range(2)]:
        summation_to_coordinate[2 ** (i[0] + 2*i[1] + 4*i[2])] = i

    templates_triangles = []
    for _ in range(256):
        templates_triangles.append( [[],[]] )

    for i in range(1,255):
        array = np.zeros((2,2,2))
        index = i
        for j in range(7,-1,-1):
            e = 2**j
            if index >= e:
                index -= e
                array[summation_to_coordinate[e]] = 1
        verts, faces = measure.marching_cubes_lewiner(array)[0:2]
        templates_triangles[i][0] = verts
        templates_triangles[i][1] = faces

    with open(MC_TEMPLATES_FILE, mode = 'w') as file:
        for i in range(256):
            verts, faces = templates_triangles[i]
            file.write(f'{i};')
            for v in verts:
                file.write(f'[{v[0]},{v[1]},{v[2]}]')
            file.write(';')
            for f in faces:
                file.write(f'[{f[0]},{f[1]},{f[2]}]')
            file.write('\n')

def create_mc_template_list(spacing = (1,1,1)):
    '''
    Return area and volume lists for the marching cubes templates
    Reads the templates file
    Input:
        Tuple with three values for x, y, and z lengths of the voxel edges
    '''
    areas = {}
    volumes = {}
    triangles = {}
    vertices_on_top = set((16,32,64,128))
    with open(MC_TEMPLATES_FILE, mode = 'r') as file:
        for line in file:
            index, verts, faces = line.split(';')
            index = int(index)
            if len(verts) > 0:
                verts = verts.strip()[1:-1].split('][')
                verts = [v.split(',') for v in verts]
                verts = [[float(edge) for edge in v] for v in verts]
                faces = faces.strip()[1:-1].split('][')
                faces = [f.split(',') for f in faces]
                faces = [[int(edge) for edge in f] for f in faces]
            else:
                verts = []
                faces = []

            occupied_vertices = set()
            sub_index = index
            for i in range(7,-1,-1):
                e = 2 ** i
                if sub_index >= e:
                    occupied_vertices.add(e)
                    sub_index -= e
            total_vertices_on_top = len(occupied_vertices & vertices_on_top)
            if total_vertices_on_top == 0:
                basic_volume = 0
            elif total_vertices_on_top == 1:
                basic_volume = 1/8
            elif total_vertices_on_top == 2:
                if ((16 in occupied_vertices and 128 in occupied_vertices) or
                    (32 in occupied_vertices and 64 in occupied_vertices)):
                    basic_volume = 1/4
                else:
                    basic_volume = 1/2
            elif total_vertices_on_top == 3:
                basic_volume = 7/8
            elif total_vertices_on_top == 4:
                basic_volume = 1

            for f in faces:
                v0, v1, v2 = [verts[i] for i in f]
                v0_proj, v1_proj, v2_proj = [(i[0],i[1],0) for i in (v0,v1,v2)]
                mean_z = sum([i[2] for i in (v0,v1,v2)])/3
                proj_area = area_of_triangle(v0_proj, v1_proj, v2_proj)
                direction = face_orientation(v0,v1,v2)
                basic_volume += mean_z * proj_area * direction

            for i in range(len(verts)):
                verts[i] = [j[0] * j[1] for j in zip(verts[i], spacing)]

            triangles[index] = (tuple(verts), tuple(faces), basic_volume)

    voxel_volume = np.prod(np.array(spacing))
    for i in triangles:
        area = 0
        verts, faces, relative_volume = triangles[i]
        for f in faces:
            triangle_area = area_of_triangle(verts[f[0]],
                                             verts[f[1]],
                                             verts[f[2]])
            area += triangle_area
        volume = voxel_volume * relative_volume
        areas[i] = area
        volumes[i] = volume

    return areas, volumes


def cube_generator():
    '''
    Generator yelds (x, y, z) coordinates for hollow cubes centered in (0, 0, 0)
    and edge length increasing by 2 each new cube.
    '''
    x = -1
    y = -1
    z = -1
    while 1:
        out = (x, y, z)

        if abs(x) == abs(y) and abs(z) <= abs(x):
            if -abs(x) <= z and z < abs(x):
                z += 1
            elif -abs(x) <= z and z == abs(x):
                if x < 0 and y < 0:
                    z = -z
                    x += 1
                elif x > 0 and y < 0:
                    z = -z
                    x = -x
                    y += 1
                elif x < 0 and y > 0:
                    z = -z
                    x += 1
                elif x > 0 and y > 0:
                    x = -z - 1
                    y = -z - 1
                    z = -z - 1
        elif abs(x) < abs(y) and z == -abs(y):
            z += 1
        elif abs(x) < abs(y) and z == abs(y):
            z = -z
            x += 1
        elif abs(x) > abs(y) and z == -abs(x):
            z += 1
        elif abs(x) > abs(y) and z == abs(x):
            z = -z
            if x < 0:
                x += 1
            elif x > 0:
                x = -x
                y += 1
        elif z < 0 and abs(x) < abs(z) and abs(y) < abs(z):
            z = -z
        elif z > 0 and abs(x) < z and abs(y) < z:
            z = -z
            x += 1
        elif abs(x) < abs(y) and abs(z) < abs(y):
            z += 1
        elif abs(y) < abs(x) and abs(z) < abs(x):
            z += 1
        else:
            print("Error: ", x, y, z)

        yield out

def check_percolation(img):
    '''
    Returns True if binary image percolates along the z axis
    '''
    labeled = ndimage.label(img)[0]
    bottom_labels = np.unique(labeled[:,:,0])
    top_labels = np.unique(labeled[:,:,-1])
    percolating_labels = np.intersect1d(
				               bottom_labels, top_labels, assume_unique = True)
    percolating_labels_total = (percolating_labels > 0).sum()

    return percolating_labels_total > 0

def remove_non_percolating(img):
    '''
    return image with non-percolating elements changed to 0
    '''
    labeled = ndimage.label(img)[0]
    bottom_labels = np.unique(labeled[:,:,0])
    top_labels = np.unique(labeled[:,:,-1])
    percolating_labels = np.intersect1d(
				               bottom_labels, top_labels, assume_unique = True)
    if percolating_labels[0] == 0:
        percolating_labels = percolating_labels[1:]


    return img * np.isin(img, percolating_labels)

def wrap_sample(img, label = -1):
    '''
    Uses convex hull to label space around a sample as -1
    '''
    if img.max() > 127:
        img = img // 2
    img = img.astype('int8')
    outside = (np.int8(1) - morphology.convex_hull_image(img))

    return img - outside



# IO functions
def load_raw(raw_file_path):
    '''
    Takes a file path for a raw image, if it has a valid file and an apropriate
    .config file, will open it, convert into a numpy array and return the array
    along the config file and the config order, used to save the output raw
    file.
    '''
    config = {}
    config_order = []
    config_path = raw_file_path[:-4] + '.config'
    with open(config_path, mode = 'r') as config_file:
        for line in config_file:
            data = [i.strip() for i in line.split('=')]
            config[data[0]] = data[1]
            config_order.append(data[0])
    data_type = 'int' + str(int(config['pixel_bytes'])*8)
    if config['signed'] == '1':
        data_type = 'u' + data_type
    img = np.fromfile(raw_file_path, dtype = data_type)
    img.resize(int(config['depth']),
               int(config['height']),
               int(config['width']))

    return img, config, config_order

def load_bmp_files(files):
    '''
    Loads and stacks .bmp files
    '''
    config = {}
    config_order = []
    img_slice = Image.open(files[0]).convert('L').transpose(Image.TRANSPOSE)
    y, x = img_slice.size
    z = len(files)
    img = np.zeros((x,y,z), dtype = 'uint8')
    img[:,:,0] = np.array(img_slice)
    i = 1
    for f in files[1:]:
        img_slice = Image.open(f).convert('L').transpose(Image.TRANSPOSE)
        img[:,:,i] = np.array(img_slice)
        i += 1

    config['width'] = x
    config_order.append('width')
    config['height'] = y
    config_order.append('height')
    config['depth'] = z
    config_order.append('depth')
    config['pixel_bytes'] = 1
    config_order.append('pixel_bytes')
    config['bigendian'] = 0
    config_order.append('bigendian')
    config['lx'] = 1.000000
    config_order.append('lx')
    config['ly'] = 1.000000
    config_order.append('ly')
    config['lz'] = 1.000000
    config_order.append('lz')
    config['unit'] = 'mm'
    config_order.append('unit')
    config['signed'] = 0
    config_order.append('signed')

    return (img, config, config_order)


def save_raw(raw_file_path, img, config, config_order):
    '''
    Saves a .raw and a .config files. File path should end in .raw
    '''

    raw_output_path = raw_file_path[:-4] + '_output.raw'
    img.tofile(raw_output_path)

    with open(raw_file_path[:-4] + '_output.config',
                                                    mode = 'w') as config_file:
        for param in config_order:
            if param == 'width':
                config_file.write('width='+str(img.shape[2])+'\n')
            elif param == 'height':
                config_file.write('height='+str(img.shape[1])+'\n')
            elif param == 'depth':
                config_file.write('depth='+str(img.shape[0])+'\n')
            elif param == 'pixel_bytes':
                p = re.compile(r'.*?([0-9]+)')
                bits = int(p.findall(str(img.dtype))[0])
                config_file.write('pixel_bytes='+str(bits//8)+'\n')
            elif param == 'signed':
                p = re.compile(r'^u', re.IGNORECASE)
                if p.search(str(img.dtype)):
                    config_file.write('signed=1\n')
                else:
                    config_file.write('signed=0\n')
            else:
                newline = param + '=' + str(config[param])
                config_file.write(newline+'\n')

#Operations
def otsu_threshold(img):

    val = filters.threshold_otsu(img)
    return (img >= val).astype('int8')

def watershed(img, compactness, two_d = False):

    if np.max(img) > 1:
        img = otsu_threshold(img)
    img[0,:,:]=0
    img[-1,:,:] = 0
    img[:,0,:] = 0
    img[:,-1,:] = 0
    if img.shape[2] >= 3:
        img[:,:,0] = 0
        img[:,:,-1] = 0
    else:
        x, y, z = img.shape
        temp_img = np.zeros((x,y,z+2))
        temp_img[:,:,1:-1] = img
        img = temp_img

    tempo = time.process_time()
    print ('Start', time.process_time()-tempo)
    tempo = time.process_time()

    if two_d:
        sampling = (1,1,1000)
    else:
        sampling = (1,1,1)

    #Calcular mapa de distância
    distance_map = ndimage.morphology.distance_transform_edt(img,
                                                          sampling = sampling)

    h, w, d = img.shape
    print ('Finished distance map', time.process_time()-tempo)
    tempo = time.process_time()

    #Identificar máxmos locais
    it = ((i,j,k) for i in range(1,h-1)
                  for j in range(1,w-1)
                  for k in range(1,d-1))
    mask = np.ones((3,3,3))
    mask[1,1,1] = 0
    markers = np.zeros_like(img).astype('uint32')
    disp_it = ((i,j,k) for i in range(-1,2) for j in range(-1,2) for k in range(-1,2))
    x, y, z = markers.shape

    for dx,dy,dz in disp_it:

        markers[1:-1,1:-1,1:-1] = np.maximum(distance_map[slice(1+dx,(-1+dx if -1+dx !=0 else None)),
                                          slice(1+dy,(-1+dy if -1+dy !=0 else None)),
                                          slice(1+dz,(-1+dz if -1+dz !=0 else None))],
                             markers[slice(1,-1),slice(1,-1),slice(1,-1)])


    markers = distance_map >= markers
    markers = markers.astype('uint32')

    print ('Finished local maxima', time.process_time()-tempo)
    tempo = time.process_time()

    #Unificar máximos agregados
    labels = ndimage.label(markers,
                           structure = ndimage.generate_binary_structure(3, 3),
                           output = markers)
    objects_box_slice = ndimage.find_objects(markers)
    print(len(objects_box_slice))
    for i in range(labels):
        sl = objects_box_slice[i]
        label = i+1
        sub_img = markers[sl]

        if sub_img.size == 1: continue

        center = [i//2 for i in sub_img.shape]

        if sub_img[tuple(center)] == label:
            sub_img *= sub_img != label
            sub_img[tuple(center)] = label
            continue

        else:
            cube_it = cube_generator()
            center = np.array(center)
            while True:
                disp = np.array(next(cube_it))
                try:
                    if sub_img[tuple(center + disp)] == label:
                        sub_img *= sub_img != label
                        sub_img[tuple(center + disp)] = label
                        break
                except IndexError:
                    pass
    print ('Finished maxima aglutinator', time.process_time()-tempo)
    tempo = time.process_time()


    it = ((i,j,k) for i in range(1,h-1)
                  for j in range(1,w-1)
                  for k in range(1,d-1))
    min_radius = int(np.mean(markers>=1 * distance_map))
    for x,y,z in it:
        if markers[x,y,z] == 0: continue
        radius = max(int(distance_map[x,y,z]),min_radius)
        sub_img = markers[x-radius:x+radius+1,
                          y-radius:y+radius+1,
                          z-radius:z+radius+1]
        marker_distance = distance_map[x,y,z]
        if np.maximum == marker_distance:
            label = markers[x,y,z]
            lower_elements = sub_img >= label
            sub_img[:,:,:] *= lower_elements

    print ('Finished maxima mask', time.process_time()-tempo)
    tempo = time.process_time()

    #Aplicar watershed

    m = distance_map.max()
    dist_img = ((-distance_map.astype('int16') + m)**2).astype('uint16')
    markers = markers.astype('int32')
    out = segmentation.watershed(dist_img,
                                 markers = markers,
                                 mask = img,
                                 compactness = 1.0)
    print ('Finished watershed', time.process_time()-tempo)
    return out.astype('uint32')

def segregator(img, relative_threshold, two_d = False):

    print(f'Segregation using {relative_threshold} threshold.')

    if 'float' in str(img.dtype):
        img = (img/np.max(img)) * 254
        img - img.astype('int8')
    if np.max(img) > 1:
        img = otsu_threshold(img)

    h, w, d = img.shape
    tempo = time.process_time()
    print ('Start', time.process_time()-tempo)
    tempo = time.process_time()

    #Calcular mapa de distância
    if two_d:
        sampling = (1,1,10000)
    else:
        sampling = None
    distance_map = ndimage.morphology.distance_transform_edt(img,
                                                          sampling = sampling)
    print ('Finished distance map', time.process_time()-tempo)
    tempo = time.process_time()

    #Calcular primeiros rotulos
    label_map, max_label = ndimage.label(img, structure = np.ones((3,3,3)))

    print ('Finished label map', time.process_time()-tempo)
    tempo = time.process_time()

    #Calcular limiar de erosao
    objects = ndimage.measurements.find_objects(label_map)

    thresholds = ndimage.measurements.labeled_comprehension(distance_map, labels = label_map,
                                                            index = np.arange(1,max_label+1), func = np.max,
                                                            out_dtype = np.float, default = None)
    thresholds = np.array(thresholds) * relative_threshold
    print ('Finished local thresholds', time.process_time()-tempo)
    tempo = time.process_time()

    #Fazer fechamento seletivo
    for i in range(max_label):
        sl = distance_map[objects[i]]
        mask = label_map[objects[i]] == (i+1)
        sl *= ((sl <= thresholds[i]) * (mask)) != 1
        sphere = morphology.ball(thresholds[i]/2)
        sl += ndimage.morphology.binary_dilation(sl,
                                                 structure=sphere,
                                                 mask= mask)

    distance_map = distance_map > 0
    eroded_img = distance_map
    label_map_2, max_label_2 = ndimage.label(eroded_img, structure = np.ones((3,3,3)))
    print ('Finished selective erosion', time.process_time()-tempo)
    tempo = time.process_time()

    #Recolocar elementos erodidos
    for i in range(max_label):
        if i in [int(j*max_label/10) for j in range(10)]:
            print (int(100*(i/max_label)+1), r'%')

        sl = objects[i]
        th = i + 1
        _, indices = ndimage.morphology.distance_transform_edt( (label_map_2[sl] * (label_map[sl] == th)) == 0, return_indices = True)

        it = ((i,j,k) for i in range(0 ,sl[0].stop - sl[0].start)
                      for j in range(0 ,sl[1].stop - sl[1].start)
                      for k in range(0 ,sl[2].stop - sl[2].start))
        dilation_map = ( (img[sl] - (label_map_2[sl] > 0))
                         * (label_map[sl] == th) ).astype('int8')

        for x, y, z in it:
            if dilation_map[x,y,z] == 0: continue
            dx, dy, dz = indices[:,x,y,z]
            label_map_2[sl][x, y, z] = label_map_2[sl][dx, dy, dz]

    print ('Finished recovering erosion', time.process_time()-tempo)
    tempo = time.process_time()

    return label_map_2

def shape_factor(img, factors):
    '''
    'volume', 'surface', 'hidraulic radius', 'equivalent diameter', 'irregularity'
    - Volume = Número de pixeis  * (Lx*Ly*Lz); unidade = [UN^3]
    - Superfície = Resultado do marching cubes; unidade = [UN^2]
    - Raio hidráulico = Volume / Superfície; unidade = [UN]
    - Diâmetro equivalente = ((6/Pi) * Volume) ^ (1/3); unidade = [UN]
    - Irregularidade = Superfície / (Pi * Diâmetro_equivalente ^2); sem unidade
    '''
    results = ''
    header = ''
    if 'volume' in factors:
        header += 'volume\t'
    if 'surface' in factors:
        header += 'surface\t'
    if 'hidraulic radius' in factors:
        header += 'hidraulic radius\t'
    if 'equivalent diameter' in factors:
        header += 'equivalent diameter\t'
    if 'irregularity' in factors:
        header += 'irregularty\t'
    for i in factors:
        if not i in ('volume', 'surface', 'hidraulic radius',
                        'equivalent diameter', 'irregularity'):
            print(f'"{i}" factor not found')

    objects = ndimage.measurements.find_objects(img)
    for i in range(0, len(objects)):
        sl = objects[i]
        label = i+1
        valid = img[sl] == label
        if min(valid.shape) <= 2:
            continue
        vol = valid.sum()
        verts, faces = measure.marching_cubes_lewiner(valid)[0:2]
        sur = measure.mesh_surface_area(verts, faces)
        eq_diam = ((6/PI) * vol) ** (0.333)
        label_eval = ''

        if 'volume' in factors:
            label_eval += str(vol)+'\t'
        if 'surface' in factors:
            label_eval += str(sur)+'\t'
        if 'hidraulic radius' in factors:
            h_r = vol / sur
            label_eval += str(h_r)+'\t'
        if 'equivalent diameter' in factors:
            e_d = ((6/PI) * vol) ** (0.333)
            label_eval += str(e_d)+'\t'
        if 'irregularity' in factors:
            irr = sur / (PI * eq_diam **2)
            label_eval += str(irr)+'\t'

        results += label_eval + '\n'

    return(header, results)

def AA_pore_scale_permeability(img):

    padded_shape = [i+2 for i in img.shape]
    padded_img = np.zeros(padded_shape, dtype = img.dtype)
    padded_img[1:-1,1:-1,1:-1] = img
    dist = ndimage.morphology.distance_transform_edt(padded_img)
    dist = dist[1:-1,1:-1,1:-1]
    dist *= 10*dist
    dist -= 0.5
    solver = ConductivitySolver(dist)
    solver.solve_laplacian(estimator_array = 'simple',
                           tol = 1e-05, maxiter = 1e04)
    return solver

def formation_factor_solver(img, substitution_array = None,
                            clay_labels = [], clay_surface_conductivity = 1.0):
    '''
    calculates formation factor on a conductivity array
    '''
    if substitution_array:
        replaced_array = np.zeros(img.shape)
        for val in substitution_array:
            replaced_array += (img == val) * substitution_array[val]

        left = slice(0,-1)
        right = slice(1,None)
        full = slice(0,None)


        for sl in ( (left, full, full),
                    (right, full, full),
                    (full, right, full),
                    (full, left, full),
                    (full, full, right),
                    (full, full, left) ):

            counter_sl = tuple([ full if i == full else
                               (right if i == left else left) for i in sl ])
            replaced_array[counter_sl] += ((replaced_array[counter_sl] > 0)
                                           * np.isin(img[sl],clay_labels)
                                           * (1-np.isin(img[counter_sl],clay_labels))
                                           * clay_surface_conductivity)

        img = replaced_array

    solver = ConductivitySolver(img)
    solver.solve_laplacian(estimator_array = 'simple',
                           tol = 1e-05, maxiter = 1e04)
    return solver


def skeletonizer(img):

    if 'float' in str(img.dtype):
        img = (img/np.max(img)) * 255
        img - img.astype('int8')

    if np.max(img) > 1:
        img = otsu_threshold(img)

    return morphology.skeletonize_3d(img)

def SB_pore_scale_permeability(img):
    pass

def labeling(img):
    return ndimage.label(img)[0]

def export_stl(img, stl_path, step_size = 8):

    if 'float' in str(img.dtype):
        img = (img/np.max(img)) * 255
        img - img.astype('int8')

    if np.max(img) > 1:
        img = otsu_threshold(img)

    print('binary img')
    vertices, faces, _, _ =  measure.marching_cubes_lewiner(img,
                                                         step_size = step_size)
    print('marching cubes')
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    print('mesh')
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]
    print('cube done')
    cube.save(stl_path)

def rescale(img, factor = 0.5):

    if img.max() <= 1:
        img *= 255

    return transform.rescale(img, factor, multichannel = False,
                  preserve_range = True, anti_aliasing = True).astype('uint8')

def marching_cubes_area_and_volume(img, spacing = (1,1,1)):

    mc_templates_generator()
    X, Y, Z = img.shape
    N = img.max()
    vertex_index_array = np.array([2**i for i in range(8)])
    vertex_index_array = vertex_index_array.reshape((2,2,2),order = 'F')
    areas = np.zeros(N+1, dtype = 'float32')
    volumes = np.zeros(N+1, dtype = 'float32')
    template_areas, template_volumes = create_mc_template_list(spacing)

    it = ((i,j,k) for i in range(X-1)
                  for j in range(Y-1)
                  for k in range(Z-1))

    for x, y, z in it:
        sub_array = img[x:x+2, y:y+2, z:z+2]
        labels = np.unique(sub_array)
        for l in labels:
            if l == 0: continue
            sub_interest = sub_array == l
            template_number = (sub_interest * vertex_index_array).sum()
            areas[l] += template_areas[template_number]
            volumes[l] += template_volumes[template_number]

    return areas, volumes

def breakthrough_diameter(img, step = 0.2):

    radius = 0
    dist = ndimage.morphology.distance_transform_edt(img)
    while check_percolation(dist > radius):
        radius += step

    return 2 * radius


def covarigram_irregular(img):

    img = wrap_sample(img)
    x, y, z =  _covariogram_irregular(img)
    return {'x_results' : x, 'y_results' : y, 'z_results' : z}


@njit(parallel = True)
def _covariogram_irregular(img):

    x, y, z = img.shape
    x_results = np.zeros(x//2, dtype = np.float64)
    y_results = np.zeros(y//2, dtype = np.float64)
    z_results = np.zeros(z//2, dtype = np.float64)

    def get_normalized_correlation(left_img, right_img):
        left_values = []
        right_values = []
        products = []
        for i in range(left_img.shape[0]):
            for j in range(left_img.shape[1]):
                for k in range(left_img.shape[2]):
                    left_val = left_img[i, j, k]
                    right_val = right_img[i, j, k]
                    if left_val == -1 or right_val == -1:
                        continue
                    left_values.append(left_val)
                    right_values.append(right_val)
                    products.append(left_val * right_val)
        if len(left_values) == 0: return None
        left_values = np.array(left_values)
        right_values = np.array(right_values)
        products = np.array(products)
        correlation = products.mean()
        product_of_expectations = left_values.mean() * right_values.mean()
        left_values.sort()
        right_values.sort()
        expectation_of_product = (left_values * right_values).mean()
        normalized_correlation = ((correlation - product_of_expectations)
                                                / (expectation_of_product - product_of_expectations))
        return normalized_correlation

    for i in prange(1, x//2):
        left_img = img[i:, :, :]
        right_img = img[:-i, :, :]
        result = get_normalized_correlation(left_img, right_img)
        if not (result is None):
            x_results[i] = result
        else:
            break

    for i in prange(1, y//2):
        left_img = img[:, i:, :]
        right_img = img[:, :-i, :]
        result = get_normalized_correlation(left_img, right_img)
        if not (result is None):
            y_results[i] = result
        else:
            break

    for i in prange(1, z//2):
        left_img = img[:, :, i:]
        right_img = img[:, :, :-i]
        result = get_normalized_correlation(left_img, right_img)
        if not (result is None):
           z_results[i] = result
        else:
            break


    return x_results, y_results, z_results


def subsampling(img, jited_func):

    img = wrap_sample(img)

    if jited_func == _jit_pore_footprint:
        img = maxi_balls(img)

    if jited_func == _jit_permeability:
        #TODO fixed
        padded_shape = [i+2 for i in img.shape]
        padded_img = np.zeros(padded_shape, dtype = img.dtype)
        padded_img[1:-1,1:-1,1:-1] = img
        dist = ndimage.morphology.distance_transform_edt(padded_img)
        dist = dist[1:-1,1:-1,1:-1]
        dist *= 10*dist
        dist -= 0.5
        img = maxi_balls(img)

    result = _subsampling(img, jited_func)
    return result

@njit(parallel = True)
def _subsampling(img, jited_func, invalid_threshold = 0.1):

    x, y, z = img.shape
    max_radius = (min((x, y, z)) - 5) // 2
    results = np.zeros(max_radius - 1, dtype = np.float64)

    for i in prange(1, max_radius):
        minimum= i
        max_x = x - i - 1
        max_y = y - i - 1
        max_z = z - i - 1
        for _ in range(100):
            center = (np.random.randint(minimum, max_x),
                      np.random.randint(minimum, max_y),
                      np.random.randint(minimum, max_z))
            j = i + 1
            view = img[center[0] - i : center[0] + j,
                              center[1] - i : center[1] + j,
                              center[2] - i : center[2] + j]
            invalids = (view == -1).sum() / view.size
            if invalids <= invalid_threshold:
                break
        else:
            continue

        result = jited_func(view)
        if not result == -1:
            results[i] = result

    return results

@njit
def _jit_porosity(img):

    invalids = 0
    pores = 0
    x, y, z = img.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if img[i, j, k] == -1:
                    invalids += 1
                elif img[i, j, k] == 1:
                    pores += 1
    return pores / (img.size - invalids)

@njit
def _jit_pore_footprint(img):

    pores_n = 0
    pores_total = 0
    x, y, z = img.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if img[i, j, k] > 0:
                    pores_n += 1
                    pores_total += img[i, j, k]
    if pores_n == 0:
        return -1
    else:
        return pores_total / pores_n


@njit
def _jit_permeability(img):

    pass
    #TODO

#Interface

class Interface():

    def __init__(self):

        self.root = tk.Tk()
        self.root.title('Porous Media Analyzer')

        self.operations = {}
        self.strings = {}

        lang_file = 'pma_' + config['language'] + '.lng'

        with open(os.path.join(LOCATION, lang_file), mode = 'r') as file:
            for line in file:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                self.strings[key] = value

        for op_name, func, preview, suffix in (
                (self.get_string('OTSU'), otsu_threshold, True, '_OTSU'),
                (self.get_string('WATERSHED'), watershed, True, '_WATE'),
                (self.get_string('AOS'), segregator, True, '_AOSE'),
                (self.get_string('SHAPE'), shape_factor, False, '_SHAP'),
                (self.get_string('AAPS'), AA_pore_scale_permeability, False, '_AAPP'),
                (self.get_string('SKEL'), skeletonizer, True, '_SKEL'),
                (self.get_string('SBPS'), SB_pore_scale_permeability, False, '_SBPP'),
                (self.get_string('LABL'), labeling, False, '_LABL'),
                (self.get_string('ESTL'), export_stl, False, '_ESTL'),
                (self.get_string('RESC'), rescale, False, '_RESC'),
                (self.get_string('MCAV'), marching_cubes_area_and_volume, False, '_MCAV'),
                (self.get_string('FFSO'), formation_factor_solver, False, '_FFSO'),
                (self.get_string('BKDI'), breakthrough_diameter, False, '_BKDI')):
            self.operations[op_name] = {
                    'function': func,
                    'preview': preview,
                    'suffix': suffix }

        operations_list = list(self.operations.keys())
        self.selected_operation = tk.StringVar(self.root)
        self.selected_operation.set(self.get_string('SELECT_OPERATION'))
        self.selected_operation.trace(
                         'w', lambda w, x, y: self.update_op_description())
        self.operations_menu = tk.OptionMenu(self.root,
                           self.selected_operation, *tuple(operations_list))
        self.operations_menu.config(width=50)
        self.operations_menu.pack(side = tk.TOP)

        self.op_description = tk.Message(self.root , width = 300)
        self.op_description.pack(side = tk.TOP)

        self.frm_main_buttons = tk.Frame(self.root)
        self.frm_main_buttons.pack(side=tk.TOP)

        self.btn_select_main = tk.Button(self.frm_main_buttons,
                                  text = self.get_string('MAIN_SELECT_BUTTON'),
                                  command = self.select_image,
                                  state = tk.DISABLED)
        self.btn_select_main.pack(side = tk.LEFT, padx = 10)
        self.btn_close_main = tk.Button(self.frm_main_buttons,
                                   text = self.get_string('MAIN_CLOSE_BUTTON'),
                                   command = self.root.destroy)
        self.btn_close_main .pack(side= tk.LEFT)

        self.lbl_extras = tk.Label(self.root, text = 'Extra functions')
        self.lbl_extras.pack(side = tk.TOP)
        self.frm_extra_buttons = tk.Frame(self.root)
        self.frm_extra_buttons.pack(side = tk.TOP)
        self.btn_convert_bmp = tk.Button(self.frm_extra_buttons,
                                         text = 'Convert BMP to RAW',
                                         command = self.convert_bmp_to_raw)
        self.btn_convert_bmp.pack(side=tk.LEFT, padx = 5)

    def update_op_description(self):
        operation = self.selected_operation.get()
        suffix = self.operations[operation]['suffix']
        description_string = self.get_string('DESCRIPTION' + suffix)
        self.op_description.config(text = description_string)
        self.btn_select_main.config(state = tk.ACTIVE)

    def close_all(self):
        self.top_preview.destroy()
        self.root.destroy()

    def select_image(self):
        self.root.withdraw()

        self.img_path = filedialog.askopenfilename()

        self.img, self.img_config, self.config_order = load_raw(self.img_path)
        self.top_preview = tk.Toplevel(self.root)
        self.top_preview.title('Preview')
        self.top_preview.protocol("WM_DELETE_WINDOW", self.close_all)
        self.cnv_preview = tk.Canvas(self.top_preview, width=200, height=200)
        self.cnv_preview.pack(side = tk.TOP)
        self.msg_preview = tk.Message(self.top_preview, width = 120)
        self.fill_text_preview(text_widget = self.msg_preview)
        self.msg_preview.pack(side = tk.TOP)
        self.dct_parameters = {}
        self.frm_preview_parameters = tk.Frame(self.top_preview)
        self.create_parameters_frame(self.frm_preview_parameters,
                                     self.dct_parameters)
        self.frm_preview_parameters.pack(side = tk.TOP)
        self.frm_preview_buttons = tk.Frame(self.top_preview)
        self.btn_preview_preview = tk.Button(self.frm_preview_buttons,
                                 text = self.get_string('BTN_PREVIEW_PREVIEW'),
                                 command = self.preview_preview)
        if not self.operations[self.selected_operation.get()]['preview']:
            self.btn_preview_preview.config(state = tk.DISABLED)
        self.btn_preview_preview.pack(side = tk.LEFT, padx = 10)
        self.btn_preview_run = tk.Button(self.frm_preview_buttons,
                             text = self.get_string('BTN_PREVIEW_RUN'),
                             command = self.preview_run)
        self.btn_preview_run.pack(side = tk.LEFT, padx = 10)
        self.btn_preview_cancel = tk.Button(self.frm_preview_buttons,
                                  text = self.get_string('BTN_PREVIEW_CANCEL'),
                                  command = self.preview_cancel)
        self.btn_preview_cancel.pack(side = tk.LEFT, padx = 10)
        self.frm_preview_buttons.pack(side = tk.TOP)
        self.preview_img = None
        self.preview_vol = None
        self.create_preview_images()

    def preview_cancel(self):
        self.top_preview.withdraw()
        self.root.iconify()

    def preview_preview(self):

        op = self.selected_operation.get()
        op_suffix = self.operations[op]['suffix']
        if op_suffix == '_OTSU':
            pre_im = otsu_threshold(self.preview_vol[:,:,0])
        elif op_suffix == '_WATE':
            compactness = float(self.dct_parameters['compactness'].get())
            pre_im = watershed(self.preview_vol, compactness, two_d = True)
            pre_im = pre_im[:,:,1]
        elif op_suffix == '_AOSE':
            threshold = float(self.dct_parameters['threshold'].get())
            pre_im = segregator(self.preview_vol, threshold, two_d = True)
            pre_im = pre_im[:,:,1]
        elif op_suffix == '_SKEL':
            pre_im = skeletonizer(self.preview_vol).sum(axis = 2)

        self.preview_img = np.array(Image.fromarray(pre_im).resize((300,300)))
        self.preview_img = (self.preview_img/np.max(self.preview_img))*254
        self.preview_img = self.preview_img.astype('int8')

        self.tk_img =  ImageTk.PhotoImage(image=
                                        Image.fromarray(self.preview_img))
        self.cnv_preview.create_image((0,0), anchor="nw", image=self.tk_img)

    def preview_run(self):
        op = self.selected_operation.get()
        op_suffix = self.operations[op]['suffix']
        self.dct_parameters


        if op_suffix == '_OTSU':
            out_img = otsu_threshold(self.img)
            save_raw(self.img_path[:-4]+op_suffix+'.raw',
                     out_img,
                     self.img_config,
                     self.config_order)

        elif op_suffix == '_WATE':
            try:
                compactness = float(self.dct_parameters['compactness'].get())
            except ValueError:
                messagebox.showinfo('Error', 'Entry is not a float')
                return
            out_img = watershed(self.img, compactness)
            save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)
        elif op_suffix == '_AOSE':
            try:
                threshold = float(self.dct_parameters['threshold'].get())
            except ValueError:
                messagebox.showinfo('Error', 'Entry is not a float')
                return
            out_img = segregator(self.img, threshold)
            save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)
        elif op_suffix == '_SHAP':
            factors = []
            for i in ('volume', 'surface', 'hidraulic radius',
                      'equivalent diameter', 'irregularity'):
                if self.dct_parameters[i].get() == 1:
                    factors.append(i)
            header, lines = shape_factor(self.img, factors)
            with open(self.img_path[:-4]+op_suffix+'.txt', mode = 'w') as file:
                file.write(header+'\n')
                file.write(lines)

        elif op_suffix == '_SKEL':
            out_img = skeletonizer(self.img)
            save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)

        elif op_suffix == '_LABL':
            out_img = labeling(self.img)
            save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)

        elif op_suffix == '_AAPP':
            permeability = AA_pore_scale_permeability(self.img)
            messagebox.showinfo('Permeability result',
                                f'Calculated permeability is {permeability.solution}')

        elif op_suffix == '_ESTL':
            save_path = self.img_path[:-4]+'.stl'
            try:
                step_size = int(self.dct_parameters['step_size'].get())
            except ValueError:
                messagebox.showinfo('Error', 'Entry is not a integer')
                return

            export_stl(self.img, save_path, step_size)

        elif op_suffix == '_RESC':

            try:
                factor = float(self.dct_parameters['factor'].get())
            except ValueError:
                messagebox.showinfo('Error', 'Entry is not a float')
                return

            out_img = rescale(self.img, factor)

            save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)

        elif op_suffix == '_MCAV':
            start = time.perf_counter()
            areas, volumes = marching_cubes_area_and_volume(self.img)
            with open(self.img_path[:-4]+op_suffix+'.txt', mode = 'w') as file:
                file.write('Index\tArea\tVolume\n')
                for i in range(1,len(areas)):
                    file.write(f'{i}\t{areas[i]}\t{volumes[i]}\n')
            print(time.perf_counter() - start)

        elif op_suffix == '_BKDI':
            start = time.perf_counter()
            step = float(self.dct_parameters['step'].get())
            diameter = breakthrough_diameter(self.img, step)
            with open(self.img_path[:-4]+op_suffix+'.txt', mode = 'w') as file:
                file.write(f'{self.img_path} - Breakthrough diameter = {diameter}')
            print(time.perf_counter() - start)

        #elif op_suffix == '_AAPP':
        #elif op_suffix == '_SBPP':
        messagebox.showinfo('Done', 'Done')
        self.top_preview.withdraw()
        self.root.iconify()

    def create_preview_images(self):
        if self.img.shape[2] > 5:
            middle_slice = self.img.shape[2]//2
            self.preview_vol = self.img[:, :, middle_slice-2 : middle_slice+3]
        else:
            self.preview_vol = self.img.copy()
        self.preview_img = np.array(Image.fromarray(
                                   self.preview_vol[:,:,0]).resize((300,300)))
        self.preview_img = (self.preview_img/np.max(self.preview_img))*254
        self.preview_img = self.preview_img.astype('uint8')

        self.tk_img =  ImageTk.PhotoImage(image=
                                        Image.fromarray(self.preview_img))
        self.cnv_preview.create_image((0,0), anchor="nw", image=self.tk_img)

    def create_parameters_frame(self, frame, dict_parameters):
        op = self.selected_operation.get()
        op_suffix = self.operations[op]['suffix']
        if op_suffix == '_OTSU':
            pass
        elif op_suffix == '_WATE':
            dict_parameters['compactness'] = tk.StringVar()
            self.lbl_param_threshold = tk.Label(frame, text = 'Compactness: ')
            self.ent_param_threshold = tk.Entry(frame,
                                textvariable = dict_parameters['compactness'])
            self.lbl_param_threshold.grid(row=0, column = 0)
            self.ent_param_threshold.grid(row=0, column = 1)
        elif op_suffix == '_AOSE':
            dict_parameters['threshold'] = tk.StringVar()
            self.lbl_param_threshold = tk.Label(frame, text = 'Threshold: ')
            self.ent_param_threshold = tk.Entry(frame,
                                textvariable = dict_parameters['threshold'])
            self.lbl_param_threshold.grid(row=0, column = 0)
            self.ent_param_threshold.grid(row=0, column = 1)
        elif op_suffix == '_SHAP':
            for i in ('volume', 'surface', 'hidraulic radius',
                      'equivalent diameter', 'irregularity'):
                dict_parameters[i] = tk.IntVar()
                dict_parameters[i].set(1)
                tk.Checkbutton(frame, text=i, variable=dict_parameters[i]).pack(side= tk.TOP)
        elif op_suffix == '_ESTL':
            dict_parameters['step_size'] = tk.StringVar()
            self.lbl_param_stepsize = tk.Label(frame, text = 'Step size: ')
            self.ent_param_stepsize = tk.Entry(frame,
                                  textvariable = dict_parameters['step_size'])
            self.lbl_param_stepsize.grid(row = 0, column = 0)
            self.ent_param_stepsize.grid(row = 0, column = 1)
        elif op_suffix == '_RESC':
            dict_parameters['factor'] = tk.StringVar()
            self.lbl_param_factor = tk.Label(frame, text = 'Rescaling factor: ')
            self.ent_param_factor = tk.Entry(frame,
                                    textvariable = dict_parameters['factor'])
            self.lbl_param_factor.grid(row = 0, column = 0)
            self.ent_param_factor.grid(row = 0, column = 1)
        elif op_suffix == '_BKDI':
            dict_parameters['step'] = tk.StringVar()
            dict_parameters['step'].set('0.1')
            self.lbl_param_factor = tk.Label(frame, text = 'Erosion step: ')
            self.ent_param_factor = tk.Entry(frame,
                                    textvariable = dict_parameters['step'])
            self.lbl_param_factor.grid(row = 0, column = 0)
            self.ent_param_factor.grid(row = 0, column = 1)
        elif op_suffix == '_FFSO':
            #TODO
            pass
        #elif op_suffix == '_AAPP':
        #elif op_suffix == '_AOSE':
        #elif op_suffix == '_SKEL':
        #elif op_suffix == '_SBPP':

    def fill_text_preview(self, text_widget):
        name = self.img_path.split('/')[-1]
        dtype = str(self.img.dtype)
        if 'int' in dtype and self.img.max() <= 1 and self.img.min() >=0:
            binary = ' (binary)'
        else:
            binary = ''
        shape = str(self.img.shape)
        text_widget.config(text=f'{name}\n{dtype} {binary}\n{shape}')

    def get_string(self, str_key):

        if str_key in self.strings:
            return self.strings[str_key]
        else:
            print('Missing string: ' + str_key)
            return str_key

    def convert_bmp_to_raw(self):
        files = filedialog.askopenfilenames()
        img, config, config_order = load_bmp_files(files)
        out_path = ''
        for i in zip(files[0], files[-1]):
            if i[0] == i[1]:
                out_path += i[0]
            else:
                break
        out_path += '.raw'
        save_raw(out_path, img, config, config_order)
        messagebox.showinfo('Done converting',
                            f'Raw image with config saved as {out_path}')


def main():
    img = np.fromfile('small_OTSU_output.raw', dtype = 'int8')
    img = img.reshape((100,100,100))
    print(subsampling(img, _jit_pore_footprint))

def run():
    interface = Interface()
    interface.root.mainloop()

if __name__ == "__main__":
    main()
