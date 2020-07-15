# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:12:55 2020

@author: Rafael Arenhart
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import PIL

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



def cube2index(img):
    return (img * indexing_template).sum()

def get_minkowsky_functionals(img):

    x, y, z = img.shape

    results = np.zeros(6, dtype=np.int64)
    for i in range(x-1):
        for j in range(y-1):
            for k in range(x-1):
                template_index = cube2index(img[i:i+2,j:j+2,k:k+2])
                minkowsky_template = isotropic_templates[template_index]
                results += minkowsky_values[minkowsky_template]
    inner_size = (x-1) * (y-1) * (z-1)
    results = (results / minkowsky_divisors)/inner_size

    return results

def subsampling(img):

    x, y, z = img.shape
    max_radius = (min((x, y, z)) - 3) // 2
    results = np.zeros((max_radius, 6))
    for i in range(1, max_radius):
        minimum= i
        max_x = x - i - 1
        max_y = y - i - 1
        max_z = z - i - 1
        center = (np.random.randint(minimum, max_x),
                      np.random.randint(minimum, max_y),
                      np.random.randint(minimum, max_z))
        j = i + 1
        view = img[center[0] - i : center[0] + j,
                          center[1] - i : center[1] + j,
                          center[2] - i : center[2] + j]

    results[i, :] = get_minkowsky_functionals(view)
    return results


def otsu_threshold(img):
    return (img.max() - img.min()) / 2
    st.write('begin otsu')
    img_min = img.min()
    img_max = img.max()
    bins = range(img_min-1, img_max+1, 2)
    hist = np.histogram(img, bins, density = True)[0]
    variances = np.zeros(len(hist))
    for i in range(1, len(variances)):
        prob_0 = hist[:i].sum()
        prob_1 = hist[:i].sum()
        if prob_0 == 0 or prob_1 == 0:
            continue
        mean_0 = (hist[:i] * bins[1:i+1]).sum() / prob_0
        mean_1 = (hist[i:] * bins[i+1:]).sum() / prob_1
        var = prob_0 * prob_1 * (mean_0 - mean_1)**2
        variances[i] = var

    max_var = 0
    t = 0
    for i, var in enumerate(variances):
        if var > max_var:
            t = i
            max_var = var
    st.text(variances)
    return t*2 + img_min


st.title('Minkowsky Functionals REV estimator')

@st.cache
def get_img_and_threshold():
    img = PIL.Image.frombytes('L', (1, 2*10**6), raw.read())
    img = np.array(img)[:,0]
    img = img.view('uint16')
    img = img.reshape(z_length, y_length, z_length)
    threshold = otsu_threshold(img)
    return img, threshold


raw = st.file_uploader('Chose raw image')
if raw:
    size = len(raw.getbuffer())
    st.text(f'Image size is {size} bytes')

    x_length = st.number_input('X length', min_value=0, value=0)
    y_length = st.number_input('Y length', min_value=0, value=0)
    z_length = st.number_input('Z length', min_value=0, value=0)
    byte_length = st.number_input('Byte length', min_value=1, value=1)
    expected_size = x_length * y_length * z_length * byte_length
    st.text(f'Image size is {expected_size} bytes')

    if size == expected_size:
        if st.button('Read and format image'):
            img, threshold = get_img_and_threshold()

    fig, axes = plt.subplots(2, 3)
    axes[0,0].imshow(img[x_length//2, :, :], cmap = 'Greys')
    axes[0,1].imshow(img[:, y_length//2, :], cmap = 'Greys')
    axes[0,2].imshow(img[:, :, z_length//2], cmap = 'Greys')
    axes[1,0].imshow(img[x_length//2, :, :] < threshold, cmap = 'Greys')
    axes[1,1].imshow(img[:, y_length//2, :] < threshold, cmap = 'Greys')
    axes[1,2].imshow(img[:, :, z_length//2] < threshold, cmap = 'Greys')
    st.write(fig)
    invert = st.checkbox('Invert colors (white is always active phase)')


    img = (img < threshold).astype(np.uint8)
    if invert:
        img = 1 - img
    results = subsampling(img)
    functionals = {a: b for b, a in enumerate(minkowsky_names) }
    chosen_functional = st.selectbox('Choose functional to display graph',
                                     minkowsky_names, index = 0)
    fig2, ax = plt.subplots()
    ax.plot(results[functionals[chosen_functional],:])
    st.write(fig2)




