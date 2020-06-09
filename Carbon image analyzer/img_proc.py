# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 17:56:46 2016

@author: Arenhart
"""

import PIL.Image as pil
import numpy as np
import scipy.ndimage as sp
import scipy.optimize as opt
from scipy import fftpack
import matplotlib.pyplot as plt
import skimage as sk
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.measure as measure
import skimage.feature as feature
import skimage.segmentation as segmentation
#import Tkinter as tk
#import tkFileDialog as filedialog
import tkinter as tk
import tkinter.filedialog as filedialog
import porespy as ps

root = tk.Tk()
root.withdraw()


def carregar():
    return pil.open(filedialog.askopenfilename())


def salvar(imagem):
    save_name = filedialog.asksaveasfilename()
    try:
        imagem.save(save_name)
    except:
        if '.' in save_name:
            save_name = save_name[:save_name.find('.')] + '.bmp'
        else:
            save_name = save_name + '.bmp'
        imagem.save(save_name)

def vis(matriz):
    return pil.fromarray(matriz)

def binarizar(matriz, limiar=None):
    if limiar == None:
        limiar = filters.threshold_otsu(matriz)
    return ((matriz >= limiar) *255).astype('uint8')
    
def histograma(matriz, bins = 20):
    return plt.hist(matriz.flatten(),bins=bins)

def mapa_distancia(matriz_binarizada):
    return sp.morphology.distance_transform_edt(matriz_binarizada)

def inverter(matriz):
    return 255 - matriz
    
def expandir_contraste(matriz):
    return sk.exposure.rescale_intensity(matriz)

def equalizar_histograma(matriz):
    return (sk.exposure.equalize_hist(matriz)*255).astype('uint8')

def filtro_gaussiano(matriz,sigma):
    return (filters.gaussian_filter(matriz,
                                    sigma=sigma)*255).astype('uint8')
def filtro_mediana(matriz,tamanho):
    return filters.median(matriz,morphology.disk(tamanho))
    
def filtro_realce(matriz, tamanho=1):
    return filters.rank.enhance_contrast(matriz,morphology.disk(tamanho))
    
def filtro_prewitt(matriz):
    return (255-filters.prewitt(matriz)*255).astype('uint8')
    
def filtro_sobel(matriz):
    return (255-filters.sobel(matriz)*255).astype('uint8')
    
def filtro_scharr(matriz):
    return (255-filters.scharr(matriz)*255).astype('uint8')
    
def erosao(matriz_binaria,tamanho=1):
    matriz_binaria = matriz_binaria//255
    return (morphology.binary_erosion(
            matriz_binaria,morphology.disk(tamanho))*255).astype('uint8')
    
def dilatacao(matriz_binaria,tamanho=1):
    matriz_binaria = matriz_binaria//255
    return (morphology.binary_dilation(
            matriz_binaria,morphology.disk(tamanho))*255).astype('uint8')
            
def abertura(matriz_binaria,tamanho=1):
    matriz_binaria = matriz_binaria//255
    return (morphology.binary_opening(
            matriz_binaria,morphology.disk(tamanho))*255).astype('uint8')
            
def fechamento(matriz_binaria,tamanho=1):
    matriz_binaria = matriz_binaria//255
    return (morphology.binary_closing(
            matriz_binaria,morphology.disk(tamanho))*255).astype('uint8')
            
def granulometria(matriz_binaria):
    matriz_binaria = matriz_binaria
    area_inicial = sum(matriz_binaria.flatten())
    raio = [0]
    area_cf = [0]
    area = [0]
    i = 1
    while area_cf[-1] < 1:
        raio.append(i)
        new_area = 1 - (sum(abertura(matriz_binaria,i).flatten())/area_inicial)
        area.append(new_area-area_cf[-1])
        area_cf.append(new_area)
        i += 1
    plt.plot(raio,area,color='blue')
    plt.plot(raio,area_cf,color='green')
    plt.show()
    print (raio,area)

def correlacao(matriz_binaria):
    matriz_binaria = matriz_binaria // 255
    comprimento = range(min(matriz_binaria.shape)//2)
    tamanho_total = matriz_binaria.shape[0]*matriz_binaria.shape[1]
    correlacao_x = []
    correlacao_y = []
    matriz = matriz_binaria.flatten()
    for i in comprimento:
        correlacao_x.append(sum(matriz * np.append(matriz[i:],matriz[:i]))/tamanho_total)
    matriz = matriz_binaria.transpose().flatten()
    for i in comprimento:
        correlacao_y.append(sum(matriz * np.append(matriz[i:],matriz[:i]))/tamanho_total)
    correlacao = (np.array(correlacao_x) + np.array(correlacao_y))/2
    plt.plot(comprimento,correlacao_x,color='blue')
    plt.plot(comprimento,correlacao_y,color='red')
    plt.plot(comprimento,correlacao,color='green')
    plt.show()
    
def correlacao_cinza(matriz, periodic_boundary = False):
    corr_arr = np.zeros((max(matriz.shape)//2,matriz.ndim), dtype = 'float64')
    
    fatias_cheias = [slice(0,matriz.shape[dim]) for dim in range(matriz.ndim)]
    for eixo in range(matriz.ndim):
        tamanho_eixo = matriz.shape[eixo]
        fatia_esquerda = list(fatias_cheias)
        fatia_direita = list(fatias_cheias)
        fatia_direita_d = list(fatias_cheias)
        fatia_esquerda_d = list(fatias_cheias)
        
        for i in range(tamanho_eixo//2):
            
            fatia_esquerda[eixo] = slice(0,i)
            matriz_esquerda = matriz[tuple(fatia_esquerda)] #segmento "menor"
            
            fatia_direita[eixo] = slice(i,tamanho_eixo)
            matriz_direita = matriz[tuple(fatia_direita)] # segmento "maior"
            
            fatia_direita_d[eixo] = slice(tamanho_eixo-i,tamanho_eixo)
            matriz_direita_deslocada = matriz[tuple(fatia_direita_d)] #"menor"
            
            fatia_esquerda_d[eixo] = slice(0,tamanho_eixo - i)
            matriz_esquerda_deslocada = matriz[tuple(fatia_esquerda_d)] #"maior"
            
            correlacao = abs(np.subtract(matriz_direita, 
                                         matriz_esquerda_deslocada,
                                         dtype = 'int16')).sum()
            if periodic_boundary:
                correlacao += abs(matriz_esquerda - 
                                  matriz_direita_deslocada).sum()
            elif i > 0: # not periodic_boundary
                correlacao /= matriz.size/matriz_esquerda.size

            corr_arr[i,eixo] = 1/correlacao
        maximum = np.max(corr_arr[1:,eixo])
        print(maximum)
        corr_arr[:,eixo] /= maximum
        corr_arr[0,eixo] = 1.
        
    return corr_arr
        
def gerar_imagem_padroes(matriz):
    pass
        
def rotular(imagem_binaria):
    return measure.label(imagem_binaria,background=0)
    
def conectividade(matriz_binaria):
    matriz_binaria = rotular(matriz_binaria)
    comprimento = range(min(matriz_binaria.shape)//2)
    tamanho_total = matriz_binaria.shape[0]*matriz_binaria.shape[1]
    conectividade_x = []
    conectividade_y = []
    matriz = matriz_binaria.flatten()
    for i in comprimento:
        matriz_deslocada = np.append(matriz[i:],matriz[:i])
        matriz_sobreposta = np.logical_and(matriz_deslocada==matriz,matriz != -1)
        conectividade_x.append(sum(matriz_sobreposta)/tamanho_total)
    matriz = matriz_binaria.transpose().flatten()
    for i in comprimento:
        matriz_deslocada = np.append(matriz[i:],matriz[:i])
        matriz_sobreposta = np.logical_and(matriz_deslocada==matriz,matriz != -1)
        conectividade_y.append(sum(matriz_sobreposta)/tamanho_total)
    conectividade = (np.array(conectividade_x) + np.array(conectividade_y))/2
    plt.plot(comprimento,conectividade_x,color='blue')
    plt.plot(comprimento,conectividade_y,color='red')
    plt.plot(comprimento,conectividade,color='green')
    plt.show()
        
def propriedades(matriz_rotulada,bins=20):
    prop = measure.regionprops(matriz_rotulada)
    perimetros = []
    areas = []
    alongamento = []
    rugosidade = []    
    for p in prop:
        if p['minor_axis_length'] == 0 : continue
        perimetros.append(p['perimeter'])
        areas.append(p['area'])
        rugosidade.append(p['perimeter']**2/(4*np.pi*p['area']))
        alongamento.append(p['major_axis_length']/p['minor_axis_length'])
    print ('Contagem: ' + str(len(perimetros)))
    print ('Perimetros (media = ' + str(np.mean(perimetros)) + ' ; desvio padrao = ' + str(np.std(perimetros)) + ')')
    plt.hist(perimetros,bins=bins)
    plt.show()
    print ('Areas (media = ' + str(np.mean(areas)) + ' ; desvio padrao = ' + str(np.std(areas))  + ')')
    plt.hist(areas,bins=bins)
    plt.show()
    print ('Alongamento (media = ' + str(np.mean(alongamento)) + ' ; desvio padrao = ' + str(np.std(alongamento))  + ')')
    plt.hist(alongamento,bins=bins)
    plt.show()
    print ('Rugosidade (media = ' + str(np.mean(rugosidade)) + ' ; desvio padrao = ' + str(np.std(rugosidade))  + ')')
    plt.hist(rugosidade,bins=bins)
    plt.show()
        
def gerar_ruido_gaussiano(matriz,desv_pad=0.01):
    return (sk.util.random_noise(matriz,var=desv_pad)*255).astype('uint8')

def gerar_ruido_snp(matriz,quantidade=0.1):
    return (sk.util.random_noise(matriz,mode='s&p',amount=quantidade)*255).astype('uint8')

def gerar_imagem_ruido(aresta,densidade):
    return (sk.util.random_noise(np.zeros((aresta,aresta)),mode='salt',amount=densidade)*255).astype('uint8')         

def extrair_bordas(matriz, mediana = 1, gaussiano = 2, realce = 2, 
                   limiar = None, mediana2 = 0,
                   janela = 100, offset = 0):
                      
    bordas = filtro_mediana(matriz,mediana)    
    bordas = filtro_gaussiano(bordas, gaussiano)    
    bordas = filtro_realce(bordas,realce)    
    bordas = filtro_scharr(bordas)    
    bordas = (filters.threshold_adaptive(bordas,janela,
                                         offset=offset)*255).astype('uint8')
    fundo  = binarizar(matriz, limiar)    
    bordas = bordas * (fundo//255)
    bordas = filtro_mediana(bordas,mediana2)

    return bordas                 
    

def segregacao_watershed(bordas, pegada = 5, limiar = 0):
                             
    
    dist = mapa_distancia(bordas)
    picos = feature.peak_local_max(dist,indices = False, 
                                   labels = bordas,
                                   footprint = np.ones((pegada,pegada)),
                                    threshold_rel = limiar)                              
    marcadores = sp.label(picos)

    rotulos = morphology.watershed(-dist, marcadores[0], mask = bordas)
    
    
    return rotulos
    

def segregacao_watershed_2(bordas, pegada = 4, limiar = 0):
                             
    
    dist = mapa_distancia(bordas)
    
    picos = ps.network_extraction.find_peaks(dist, r = pegada)
    picos = ps.network_extraction.trim_saddle_points(picos, dist, max_iters=10)
    picos = ps.network_extraction.trim_nearby_peaks(picos, dist)                          
    
    
    marcadores = sp.label(picos)

    rotulos = morphology.watershed(-dist, marcadores[0], mask = bordas)
    
    
    return rotulos   
    
def filtro_fourier(arr, kf = 0.1):
    
    
    im_fft = fftpack.fft2(arr)
    keep_fraction = kf
    im_fft2 = im_fft.copy()
    r, c = im_fft2.shape
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    im_new = fftpack.ifft2(im_fft2).real.astype('uint8')
    
    return im_new

    

#c = correlacao_cinza(arr)
    
'''
Operacoes sobre a imagem:
redimensionar:
    imagem.resize((tamanho_x,tamanho_y))
    
cortar:
    imagem = imagem.crop((coordenada_x_esquerda,
							   coordenada_y_superior,
							   coordenada_x_direita,
							   coordenada_y_inferior))
          
converter para escala de cinza:
    imagem = imagem.convert('L')
    
separar canais de cor:
    imagem_vermelha, imagem_verde, imagem_azul = imagem.split()
    
converter imagem em matriz:
    imagem_mat = np.array(imagem)
    
converter matriz em imagem:
    imagem = pil.fromarray(imagem_mat)

'''