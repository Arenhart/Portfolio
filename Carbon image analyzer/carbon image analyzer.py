# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:20:50 2018

@author: Rafael Arenhart
"""

import os
import math
import PIL.Image as pil
import numpy as np
import skimage.filters as filters
import skimage as sk
import skimage.morphology as morphology
import skimage.measure as measure
import skimage.feature as feature
import sklearn.cluster as cluster
import tkinter.filedialog as filedialog
import img_proc as imp
from IPython.display import Image, display


LINE_THICKNESS = 3


class ImageStacker():
    
    def __init__(self, pore_image = None, carbon_image = None,
                 scan_length = 50,
                 stretch_threshold = 5,
                 subdivision = (5,5)):
        
        self.subdivision = subdivision
        self.scan_length = scan_length
        self.stretch_threshold = stretch_threshold
        
        if pore_image == None:
            pore_image = imp.carregar()
            
        if carbon_image == None:
            carbon_image = imp.carregar()
            
        self.pore_img = np.array(pore_image.copy().convert('L'))
        self.carbon_img = np.array(carbon_image.copy().convert('L'))
            
        if self.pore_img.shape != self.carbon_img.shape:
            target_size = [None,None]
            target_size[0] = min(self.pore_img.shape[0],
                                  self.carbon_img.shape[0])
            target_size[1] = min(self.pore_img.shape[1],
                                  self.carbon_img.shape[1])
            
            if self.pore_img.shape != target_size:
                self.pore_img = pil.fromarray(self.pore_img)
                self.pore_img = self.pore_img.resize(target_size,
                                                 resample=pil.BICUBIC)
                self.pore_img = np.array(self.pore_img)
                
            if self.carbon_img.shape != target_size:
                self.carbon_img = pil.fromarray(self.carbon_img)
                self.carbon_img = self.carbon_img.resize(target_size,
                                                 resample=pil.BICUBIC)
                self.carbon_img = np.array(self.carbon_img)
        
        self.offset = [0,0]
        self.stretch = [0,0]
        self.pore_edge = None
        self.carbon_edge = None
        self.pore_adjusted = None
        self.carbon_adjusted = None
        self.adjusted_edges = None
        self.output_img = None
        self.out_edge_img = None
        self.original_edges = None
        self.variance_map = None
    
    def set_edges(self):
        
        self.pore_edge, self.carbon_edge = self._extract_edge_canny(
                                                            self.pore_img, 
                                                            self.carbon_img)
        
        disk = morphology.disk(LINE_THICKNESS)
        self.pore_edge = morphology.binary_dilation(self.pore_edge, disk)
        self.carbon_edge = morphology.binary_dilation(self.carbon_edge, disk)
        
        return 1
    
    def inspect_edges(self):
        
        max_y = max(self.carbon_edge.shape[1], self.pore_edge.shape[1])
        center_line = np.ones((LINE_THICKNESS,max_y))
        all_edges = np.concatenate((self.pore_edge,
                                   center_line,
                                   self.carbon_edge), axis = 0)
        all_images = np.concatenate((self.pore_img,
                                   center_line,
                                   self.carbon_img), axis = 0)
        
        edges_img = pil.fromarray(all_edges*255)
        edges_img.show()
        all_img = pil.fromarray(all_images*255)
        all_img.show()
        
        return 1
    
    def set_adjustments(self):
        
        self._find_adjustments()
        
        self.pore_adjusted, self.carbon_adjusted = self._get_adjusted_images(
                                                self.pore_img, self.carbon_img)
        
        
        pore_edge_adj, carbon_edge_adj = self._get_adjusted_images(
                                             self.pore_edge, self.carbon_edge)
        arr = np.ones(pore_edge_adj.shape+(3,),dtype='uint8') * 255
        arr[:,:,0] = carbon_edge_adj * 255
        arr[:,:,1] = pore_edge_adj * 255
        self.out_edge_image = pil.fromarray(arr)
        
        
        arr = np.ones(self.pore_edge.shape+(3,),dtype='uint8') * 255
        arr[:,:,0] = self.carbon_edge * 255
        arr[:,:,1] = self.pore_edge * 255
        self.original_edges = pil.fromarray(arr)
        
        img1 = pil.fromarray(self.carbon_adjusted)
        img2 = pil.fromarray(self.pore_adjusted)
        
        im = pil.blend(img1.convert('RGB'),
                         img2.convert('RGB'),
                         0.5)
        self.output_img = im

        
        '''
        target_size = (self.carbon_edge.shape[1] - self.stretch[1],
                       self.carbon_edge.shape[0] - self.stretch[0])
        
        self.carbon_adjusted = pil.fromarray(self.carbon_edge.astype('uint8'))
        self.carbon_adjusted = self.carbon_adjusted.resize(target_size, 
                                                   resample = pil.BILINEAR)
        self.carbon_adjusted = np.array(self.carbon_adjusted)
        
        self.pore_adjusted = self.pore_edge.copy()
        
        left_cut = math.floor(self.offset[0] - self.stretch[0]/2)
        right_cut = math.ceil(-self.offset[0] - self.stretch[0]/2)
        up_cut = math.floor(self.offset[1] - self.stretch[1]/2)
        down_cut = math.ceil(-self.offset[1] - self.stretch[1]/2)
        
        if left_cut > 0:
            self.carbon_adjusted = self.carbon_adjusted[left_cut:,:]
        elif left_cut < 0:
            self.pore_adjusted = self.pore_adjusted[-left_cut:,:]
            
        if right_cut > 0:
            self.carbon_adjusted = self.carbon_adjusted[:-right_cut,:]
        elif right_cut < 0:
            self.pore_adjusted = self.pore_adjusted[:right_cut,:]
        
        if up_cut > 0:
            self.carbon_adjusted = self.carbon_adjusted[:,up_cut:]
        elif up_cut < 0:
            self.pore_adjusted = self.pore_adjusted[:,-up_cut:]
        
        if down_cut > 0:
            self.carbon_adjusted = self.carbon_adjusted[:,:-down_cut]
        elif down_cut < 0:
            self.pore_adjusted = self.pore_adjusted[:,:down_cut]
        '''
            
        return 1
    
    
    def inspect_adjustments(self):
        
        '''
        arr = np.ones(self.pore_adjusted.shape+(3,),dtype='uint8') * 255
        arr[:,:,0] = self.carbon_adjusted * 255
        arr[:,:,1] = self.pore_adjusted * 255
        im = pil.fromarray(arr)
        '''
        arr1 = self.carbon_adjusted.copy().astype('uint16')
        arr2 = self.pore_adjusted.astype('uint16')
        arr3 = np.zeros(list(arr1.shape)+[3,], dtype = 'uint16')
        arr3[:,:,0] = (arr1 * arr1 + arr2 * (255-arr1)) // 255
        for i in (1,2):
             arr3[:,:,i] = (arr2 * (255-arr1)) // 255
        arr3 = arr3.astype('uint8')	
			
        
        im = pil.fromarray(arr3)
        #im = self.output_img
        im.show()
        self.out_edge_image.show()
        self.original_edges.show()
        #pil.fromarray(self.variance_map.astype('uint8')).show()
        
        return 1
    
    def analyze(self, id_pores = False, sample = 10,
                      output_name ='output.txt'):
        '''
        id_pores > 0 to use Kmeans to identify pores
        id_pores is the number of clusters
        '''
        
        rot_arr = self._rotulate(self.pore_adjusted)
        self.rot_arr = rot_arr
        intensity = self.carbon_adjusted
        regionprops = measure.regionprops(rot_arr, intensity)
        medidas = [dict(zip(list(rp),[rp[i] for i in list(rp)])) 
                                    for rp in regionprops]
        #medidas = measure.regionprops(rot_arr, intensity)
        propriedades= ['label', 'area', 'eccentricity', 'mean_intensity',
                       'perimeter', 'solidity']
        output = ''
        
        if id_pores:
            propriedades += ['cluster','avg_color']
            km = cluster.KMeans(n_clusters = id_pores)
            X = []
            for m in medidas:
                box = m['bbox']
                mask = m['image']
                view = np.array(self.image1)[box[0]:box[2],box[1]:box[3]]
                pore = mask * view
                avg_color = pore.sum()/mask.sum()
                X += (avg_color,)
            km.fit(np.array(X).reshape(-1,1))
            for i in zip(medidas, km.labels_, X):
                i[0]['cluster'] = i[1]
                i[0]['avg_color'] = i[2]
                
        print(propriedades)
        for p in propriedades:
            output += f'{p}\t'
        for m in medidas:
            output +='\n'
            for p in propriedades:
                output += f'{m[p]}\t'
        
        
        
        f = open(output_name,'w')
        f.write(output)
        f.close()
        for label in [i for i in range(min(len(medidas), sample))]:
            box = medidas[label]['bbox']
            region1 = medidas[label]['image']
            region2 = np.array(self.output_img)[box[0]:box[2],box[1]:box[3]]
            region3 = medidas[label]['convex_image']
            img1 = pil.fromarray(region1.astype('uint8')*255)
            img2 = pil.fromarray(region2.astype('uint8'))
            img3 = pil.fromarray(region3.astype('uint8')*255)
            img1.save(f'bin_pore_{label}.bmp')
            img1.close()
            img2.save(f'pore_{label}.bmp')
            img2.close()
            img3.save(f'convex_pore_{label}.bmp')
            img3.close()
        

        seg_arr = np.array(self.output_img)
        for m in medidas:
            box = m['bbox']
            mask = np.zeros(seg_arr.shape[:2]).astype('uint8')
            mask[box[0]:box[2],box[1]:box[3]] = 1
            mask[box[0]+LINE_THICKNESS:box[2]-LINE_THICKNESS,
                 box[1]+LINE_THICKNESS:box[3]-LINE_THICKNESS] = 0
            seg_arr[:,:,0] *= (1 - mask)
            seg_arr[:,:,1] *= (1 - mask)
            seg_arr[:,:,2] *= (1 - mask)

            
                
        seg_img = pil.fromarray(seg_arr)
        seg_img.save('identified_pores.bmp')
        seg_img.close()
        self.medidas = medidas
        
        return 1
    
    def _find_adjustments(self):
        
        '''
        Adjustments are always relative to original image
        '''
        
        nx, ny = self.subdivision
        
        subs = [(i,j) for i in range(nx) 
                      for j in range(ny)]
        
        offsets = np.zeros((self.subdivision)+(2,))
        
        
        for sx, sy in subs:
            
            s1_x0 = int((self.pore_edge.shape[0]*sx)/nx)
            s1_x1 = int((self.pore_edge.shape[0]*(sx+1))/nx)
            s1_y0 = int((self.pore_edge.shape[1]*sy)/ny)
            s1_y1 = int((self.pore_edge.shape[1]*(sy+1))/ny)
            
            arr1 = self.pore_edge[s1_x0:s1_x1, s1_y0:s1_y1]
            
            s2_x0 = int((self.carbon_edge.shape[0]*sx)/nx)
            s2_x1 = int((self.carbon_edge.shape[0]*(sx+1))/nx)
            s2_y0 = int((self.carbon_edge.shape[1]*sy)/ny)
            s2_y1 = int((self.carbon_edge.shape[1]*(sy+1))/ny)
            
            arr2 = self.carbon_edge[s2_x0:s2_x1, s2_y0:s2_y1]
            
            off = self._find_offset(arr1, arr2)
            offsets[sx,sy,:] = off
        
        self.variance_map = off
        x_distortion = self._linear_fit_angle(offsets[:,:,0])
        y_distortion = self._linear_fit_angle(offsets[:,:,1])
        
        x_stretch = int( x_distortion * ( nx / (nx-1) ) )
        y_stretch = int( y_distortion * ( ny / (ny-1) ) )
        
        self.offset = [offsets[:,:,0].mean() + self.offset[0],
                       offsets[:,:,1].mean() + self.offset[1]]
        
        self.stretch = [x_stretch + self.stretch[0],
                        y_stretch + self.stretch[1]]
            
        
    def _get_adjusted_images(self, input1, input2, 
                             offset = None, 
                             stretch = None):
        if offset == None:
            offset = self.offset
        if stretch == None:
            stretch = self.stretch
            
        target_size = (input2.shape[1] - self.stretch[1],
                       input2.shape[0] - self.stretch[0])
        
        arr2 = pil.fromarray(input2.astype('uint8'))
        arr2 = arr2.resize(target_size, resample = pil.BILINEAR)
        arr2 = np.array(arr2)
        
        arr1 = input1.copy()
        
        left_cut = math.floor(offset[0] - stretch[0]/2)
        right_cut = math.ceil(-offset[0] - stretch[0]/2)
        up_cut = math.floor(offset[1] - stretch[1]/2)
        down_cut = math.ceil(-offset[1] - stretch[1]/2)
        
        if left_cut > 0:
            arr2 = arr2[left_cut:,:]
        elif left_cut < 0:
            arr1 = arr1[-left_cut:,:]
            
        if right_cut > 0:
            arr2 = arr2[:-right_cut,:]
        elif right_cut < 0:
            arr1 = arr1[:right_cut,:]
        
        if up_cut > 0:
            arr2 = arr2[:,up_cut:]
        elif up_cut < 0:
            arr1 = arr1[:,-up_cut:]
        
        if down_cut > 0:
            arr2 = arr2[:,:-down_cut]
        elif down_cut < 0:
            arr1 = arr1[:,:down_cut]
            
        return (arr1, arr2)
            
    
    def _linear_fit_angle(self, data):
        
        if data.ndim > 1:
            data = data.sum(axis=0)
        
        total_sum = 0
        
        for i in range(1,data.size):
            total_sum += (data[i] - data[0]) / i
            
        return total_sum / (data.size - 1)
        
    
    def _find_offset(self, arr1, arr2):
        
        '''
        Find ideal displacement of arr2 in relation to arr1
        Ex.: Offset of (10,0) meand arr2 is shifted 10 pixels to the "left"
        '''

        length = self.scan_length
        variances = np.zeros((2*length+1, 2 * length+1))
        
        for offset in [(i,j) for i in range(-length,length+1) 
                             for j in range(-length,length+1)]:
            i, j = offset
            
            if i < 0:
                off_arr1 = arr1[-i:]
                off_arr2 = arr2[:i]
            elif i == 0:
                off_arr1 = arr1
                off_arr2 = arr2    
            elif i > 0:
                off_arr1 = arr1[:-i]
                off_arr2 = arr2[i:]
                
            if j < 0:
                off_arr1 = off_arr1[:,-j:]
                off_arr2 = off_arr2[:,:j]
            elif j == 0:
                pass    
            elif j > 0:
                off_arr1 = off_arr1[:,:-j]
                off_arr2 = off_arr2[:,j:]
            
            variance_array = off_arr1 != off_arr2
            variance = variance_array.sum()/variance_array.size
            variances[i+length,j+length] = variance
            
        offset = list(np.unravel_index(variances.argmin(),
                                       variances.shape))
        
        return (offset[0] - length, offset[1] - length)  
    
    def _extract_edge_standard(self, arr,
                      pre_opening_size = 1,
                      median_filter = 3,
                      threshold = 150,
                      opening_size = 0,
                      minimum_size = 50,
                      dilation_size = 0):
        
        arr = imp.expandir_contraste(arr)
        arr = morphology.opening(arr,morphology.disk(pre_opening_size))
        arr = imp.filtro_mediana(arr,median_filter)
        arr = imp.binarizar(arr, threshold)
        arr = (255-filters.scharr(arr)*255).astype('uint8')
        arr = 255 - pil.abertura(255 - arr, tamanho = opening_size)
        arr = morphology.remove_small_holes(arr, min_size = minimum_size)
        arr = 255 - pil.dilatacao(255 - arr, tamanho = dilation_size)
        
        return arr

    def _extract_edge_canny(self, arr1, arr2,
                           sigma = 0.2,
                           pre_opening = 3):
        '''
        as in https://www.pyimagesearch.com/2015/04/06/
        zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
        '''
        
        disk = morphology.disk(pre_opening)
        
        arr1 = imp.expandir_contraste(arr1)
        arr2 = imp.expandir_contraste(arr2)
        
        arr1 = morphology.opening(arr1, disk)
        arr2 = morphology.opening(arr2, disk)
        
        median1 = np.mean(arr1)
        
        lower1 = int(max(0, (1.0 - sigma) * median1))
        upper1 = int(min(255, (1.0 + sigma) * median1))
        
        edge1 = feature.canny(arr1, 
                             low_threshold = lower1, 
                             high_threshold = upper1)
        
        density = edge1.sum()/edge1.size
        
        median2 = np.mean(arr2)
        
        lower2 = int(max(0, (1.0 - sigma) * median2))
        upper2 = int(min(255, (1.0 + sigma) * median2))
        
        edge2 = feature.canny(arr2, 
                             low_threshold = lower2, 
                             high_threshold = upper2)
        
        while edge2.sum()/edge2.size <= density:
            upper2 -= 5
            if upper2 < 0:
                upper2 = 0
                lower2 -= 5
            edge2 = feature.canny(arr2, 
                                 low_threshold = lower2, 
                                 high_threshold = upper2)
            
        while edge2.sum()/edge2.size > density:
            upper2 += 5
            if upper2 > 255: 
                upper2 = 255
                lower2 += 5 
            edge2 = feature.canny(arr2, 
                                 low_threshold = lower2, 
                                 high_threshold = upper2)
            
        return edge1, edge2
    
    def _rotulate(self, image, tamanho = 3, min_size = 50,
                 window_size=51, k=0.5):
        arr = np.array(image)
        arr = filters.rank.enhance_contrast(
                                            arr,morphology.disk(tamanho))
        #limiar = filters.threshold_otsu(arr)
        #arr = ((arr >= limiar) *255).astype('uint8')
        t_sauvola = filters.threshold_sauvola(arr, 
                                              window_size=window_size, k=k)
        arr = arr > t_sauvola
        arr = 255 - imp.abertura(255 - arr, tamanho = 1)
        arr = morphology.remove_small_holes((arr==255), min_size = 200)
        arr = arr.astype('uint8')*255
        arr = measure.label(arr,background=255)
        return arr
    
    
            

def teste():

    global stacker
    
    poros = pil.open('P4TA-IG_Pista1_500X_BSE(2)_bin.bmp')
    #poros = pil.open('poro_corte.bmp')
    carbono = pil.open('P4TA_MAPA_C_pista2_500X(2).png')
    #carbono = pil.open('carbono_corte.bmp')
    
    stacker = ImageStacker(poros, carbono)
    
    print(stacker.set_edges() == 1)
    print(stacker.inspect_edges() == 1)
    
    print(stacker.set_adjustments() == 1)
    print(stacker.inspect_adjustments() == 1)
    
    print(stacker.analyze())
    
    #results = stacker.get_results()
    #print(results)
    
    #results['variance map']
    #results['stacked image']
    #results['adjusted pore image']
    #results['adjusted carbon image']

def analisar_imagens():
    
    imagens_poro = filedialog.askopenfilenames()
    imagens_carbono = filedialog.askopenfilenames()
    
    for poro_fp, carbono_fp in zip(imagens_poro, imagens_carbono):
        poros = pil.open(poro_fp)
        carbono = pil.open(carbono_fp)
        stacker = ImageStacker(poros, carbono)
        print(stacker.set_edges() == 1)
        print(stacker.set_adjustments() == 1)
        out_name = poro_fp[:-4] + '.txt'
        print(stacker.analyze(output_name = out_name))
		
    return stacker

os.chdir('D:\Scripts Python\Imagens\Nicolas')
teste()  
#stack = analisar_imagens()  
    
    
    
    
    
    
    
    