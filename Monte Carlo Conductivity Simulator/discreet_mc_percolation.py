# -*- coding: utf-8 -*-
'''
Created on 02/12/2010

@author: rafael
'''

#from scipy.stats import norm
#import time
import time
import sys
import numpy as np
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

START_GUI = True
SUB_ARRAY_SIZE = 4
MARGIN_TOLERANCE = 1.5
PI = np.pi
TEST_SIZE = (100,)*3
SCANNING_REPEATS = 3
BOUNDS_THRESHOLD = 0.1
FAST_AUTO = [
        (100, 50),
        (200, 25),
        (300, 10)]
THRESHOLD_BOUNDS = [0, 1]
EXPONENT_BOUNDS = [-np.inf, 1]

'''
FAST_AUTO = [
        (100, 10),
        (200, 5)]
'''
SLOW_AUTO = [
        (100, 100),
        (200, 50),
        (300, 20),
        (400, 12),
        (500, 6)]

def percolation_function(x, percolation_threshold, t):
    
    is_over_threshold = (x > percolation_threshold)
    out = np.zeros((x.shape))
    
    out[is_over_threshold] = (x[is_over_threshold]-percolation_threshold) ** t
    
    return out

def intercalate(lists):
    
    weighted_lists= []
    for ls in lists:
        weighted_list = [ ((i+1)/(len(ls)+1), ls[i]) for i in range(len(ls))]
        weighted_lists.extend(weighted_list)
        
    sorted_lists = sorted(weighted_lists, key=lambda x: x[0])
    
    return [i[1] for i in sorted_lists]

class Simulacao(object):
    
    def __init__(self, concentration = 0.5,
                       shape = (100,100,100), 
                       modo_distribuicao = 'fibras',
                       parametros = {'logmean': 0.81,
                                     'logstd': 0.61}):
        '''
        modo_distribuicao deve ser 'fibras', 'pontos', 'esferas'
        '''
        self.concentration = concentration
        self.shape = shape
        self.modo_distribuicao = modo_distribuicao
        self.parametros = parametros

    def simular_percolacao(self):
        
        
        self.matriz = None
        
        if self.modo_distribuicao == 'fibras':
            
            
            self.matriz = np.zeros(self.shape)
            logmean = self.parametros['logmean']
            logstd = self.parametros['logstd']
            target_conductors = self.matriz.size * self.concentration
            current_conductors = 0
            
            while current_conductors <= target_conductors:
                
                length_fiber = np.random.lognormal(logmean,logstd)
                min_coord = -int(length_fiber)
                max_coords = [(i + int(length_fiber)) for i in self.matriz.shape] 
                fiber_start = [np.random.randint(min_coord, max_coord) for
							               max_coord in max_coords]

                theta = np.random.random() * 2 * PI
                rho = np.arccos(2*np.random.random() - 1)
                '''
                delta_x = int(np.rint(np.cos(theta) * np.cos(rho) 
                                                              * length_fiber))
                delta_y = int(np.rint(np.sin(theta) * np.cos(rho) 
                                                              * length_fiber))
                delta_z = int(np.rint(np.sin(rho) * length_fiber))
                
                displacements = []
                
                for delta, disp_array in ((delta_x, np.array((1,0,0))),
                                          (delta_y, np.array((0,1,0))),
                                          (delta_z, np.array((0,0,1)))):
                
                    if delta >= 0:
                        displacement = [disp_array,] * delta
                    else: #delta < 0
                        displacement = [disp_array * -1,] * abs(delta)
                    displacements.append(displacement)
                
                ordered_displacements = intercalate(displacements) #also flattens
                ordered_displacements.insert(0, np.array((0,0,0)))
                
                current_coord = np.array(fiber_start)
                    
                for disp in ordered_displacements:
                    
                    current_coord += disp
                    if (current_coord >= 0).all():
                        try:
                            self.matriz[tuple(current_coord)] = 1
                            current_conductors += 1
                        except IndexError:
                            pass
                '''
                
                x, y, z = fiber_start
                max_x, max_y, max_z = self.matriz.shape
                propz = np.sin(rho)
                if propz < 0 :
                    dirz = -1
                    propz = -propz
                else:
                    dirz = 1
                    
                propy=np.sin(theta)*np.cos(rho)
                if propy < 0 :
                    diry = -1
                    propy = -propy
                else:
                    diry = 1
                    
                propx=np.cos(theta)*np.cos(rho)
                if propx < 0 :
                    dirx = -1
                    propx = -propx
                else:
                    dirx=1
                
                    
                totalpoints=length_fiber * (propz+propy+propx)
    
                xacc = 0
                yacc = 0
                zacc = 0
                
                if x >= 0 and x < max_x and y >= 0 and y < max_y and z >= 0 and z < max_z and current_conductors <= target_conductors:
                    self.matriz[x,y,z] = 1
                    current_conductors += 1
                
                for i in range(int(totalpoints)+1):
                    xacc=xacc+propx
                    yacc=yacc+propy
                    zacc=zacc+propz
                    
                    if xacc >= yacc and xacc >= zacc:
                        x = x + dirx
                        xacc = 0
                    elif yacc>=zacc :
                        y = y + diry
                        yacc = 0
                    else :
                        z = z + dirz
                        zacc = 0
                    
                    if x >= 0 and x < max_x and y >= 0 and y < max_y and z >= 0 and z < max_z and current_conductors <= target_conductors:
                        self.matriz[x,y,z] = 1
                        current_conductors += 1

        elif self.modo_distribuicao == 'pontos':
            self.matriz = np.random.binomial(1, self.concentration, self.shape)
            
        elif self.modo_distribuicao == 'esferas':
            
            self.matriz = np.zeros(self.shape)
            logmean = self.parametros['logmean']
            logstd = self.parametros['logstd']
            target_conductors = self.matriz.size * self.concentration
            current_conductors = 0
            
            while current_conductors < target_conductors:
                
                diam_esfera = np.random.lognormal(logmean, logstd)
                raio_esfera_sq = (diam_esfera/2)**2
                vol_esfera = (1/6) * PI * diam_esfera**3
                
                if vol_esfera > (target_conductors - current_conductors):
                    max_vol = (target_conductors - current_conductors)
                    diam_esfera = int(np.rint((max_vol * (6/PI)) ** (1/3)))
                
                min_coord = -int(diam_esfera/2)
                max_coords = [(i + int(diam_esfera/2)) for i in self.matriz.shape] 
                center = [np.random.randint(min_coord, max_coord) for
                                                       max_coord in max_coords]
                x, y, z = center
                
                min_iterator = int(-round(diam_esfera/2)-1)
                max_iterator = int(round(diam_esfera/2)+2)
                
                for i, j, m in ((a,b,c) 
                                  for a in range(min_iterator, max_iterator)
                                  for b in range(min_iterator, max_iterator)
                                  for c in range(min_iterator, max_iterator)):
                    
                    if not (x+i >= 0 and x+i < self.shape[0] 
                        and y+j >= 0 and y+j < self.shape[1]
                        and z+m >= 0 and z+m < self.shape[2] 
                        and current_conductors < target_conductors):
                        # voxel não está na matriz
                        continue
                    
                    if not (self.matriz[x+i, y+j, z+m] == 0):
                        # voxel já é condutor
                        continue
                    
                    if (i**2 + j**2 + m**2) <= raio_esfera_sq:
                        # voxel dentro da esfera
                        self.matriz[round(x+i),round(y+j),round(z+m)] = 1
                        current_conductors += 1
    
        else:
            print(f'{self.modo_distribuicao} modo de distribuição não encontrado')
            raise
        
        self.labeled = ndimage.label(self.matriz)[0]
        
        labels = {}
        labels['x_bottom'] = set(np.unique(self.labeled[0,:,:]))
        labels['x_top'] = set(np.unique(self.labeled[-1,:,:]))
        labels['y_bottom'] = set(np.unique(self.labeled[:,0,:]))
        labels['y_top'] = set(np.unique(self.labeled[:,-1,:]))
        labels['z_bottom'] = set(np.unique(self.labeled[:,:,0]))
        labels['z_top'] = set(np.unique(self.labeled[:,:,-1]))
        
        for label in labels:
            if 0 in labels[label]: labels[label].remove(0)
            
        percolating_labels = []
        
        percolating_labels.append(labels['x_bottom'] & labels['x_top'])
        percolating_labels.append(labels['y_bottom'] & labels['y_top'])
        percolating_labels.append(labels['z_bottom'] & labels['z_top'])
        
        self.percolations = []
        for labels in percolating_labels:
            self.percolations.append(len(labels) > 0)
        
        self.interconectivities = []
        for labels in percolating_labels:
            self.interconectivities.append(
                          np.isin(self.labeled, list(labels)).sum()/self.matriz.size)
            
        return self.percolations, self.interconectivities
        
def scan_percolation(modo_distribuicao = 'fibras',
                      parametros = {'logmean' : 1.0,
                                    'logstd' : 0.5}):
    '''
    Performs a binary search of the possible lower and upper bounds for the
    percolation threshold
    
    Args:
        modo_distribuicao (string) - The shape of the conductive particles, valid values
        are 'fibras', 'pontos' and 'esferas'
        parametros (dict) - A dictionary containing the distribution parmeters of the particle
        
    Return:
        list - A list containg two floats, representing lower and upper bound of the 
        percolation threshold
    '''
    
    percolation_bounds = np.array((0, 0.5))
    fuzzy_range = np.array((None,None))
    iterations = 0
    
    while True:
        
        if (fuzzy_range == None).sum() == 2:
            test_concentration = percolation_bounds.mean()
            
        elif (fuzzy_range == None).sum() == 1:
            fuzzy_value = fuzzy_range[fuzzy_range != None][0]
            lower_range = fuzzy_value - percolation_bounds[0]
            upper_range = percolation_bounds[1] - fuzzy_value
            if lower_range >= upper_range:
                test_concentration = lower_range/2 + percolation_bounds[0]
            else:
                test_concentration = upper_range/2 + fuzzy_value
                
        else:
            lower_range = fuzzy_range[0] - percolation_bounds[0]
            upper_range = percolation_bounds[1] - fuzzy_range[1]
            if lower_range >= upper_range:
                test_concentration = lower_range/2 + percolation_bounds[0]
            else:
                test_concentration = upper_range/2 + fuzzy_range[1]
                
        percolations = 0
        
        for _ in range(SCANNING_REPEATS):
            simulador = Simulacao(test_concentration,
                                  shape = TEST_SIZE,
                                  modo_distribuicao = modo_distribuicao,
                                  parametros = parametros)
            percolations += sum(simulador.simular_percolacao()[0])
            
        if percolations == 0:
            percolation_bounds[0] = test_concentration
            
        elif percolations == (SCANNING_REPEATS * 3):
            percolation_bounds[1] = test_concentration
            
        else:
            if (fuzzy_range == None).sum() == 2:
                fuzzy_range[0] = test_concentration
            
            elif (fuzzy_range == None).sum() == 1:
                fuzzy_value = fuzzy_range[fuzzy_range != None][0]
                if fuzzy_value >= test_concentration:
                    fuzzy_range[0] = test_concentration
                    fuzzy_range[1] = fuzzy_value
                else:
                    fuzzy_range[0] = fuzzy_value
                    fuzzy_range[1] = test_concentration
            
            elif (fuzzy_range == None).sum() == 0:
                if test_concentration < fuzzy_range[0]:
                    fuzzy_range[0] = test_concentration
                else:
                    fuzzy_range[1] = test_concentration

        
        if (fuzzy_range == None).sum() == 0:
            relative_bounds = ((percolation_bounds[1]-fuzzy_range[1]) + (fuzzy_range[0] - percolation_bounds[0])) / percolation_bounds.mean()
        
        else:
            relative_bounds = ((percolation_bounds[1]) - (percolation_bounds[0])) / percolation_bounds.mean()
        
                
        print(iterations, relative_bounds, test_concentration, percolation_bounds, fuzzy_range)
        iterations += 1
        if relative_bounds <= BOUNDS_THRESHOLD:
            break
        
    return percolation_bounds
        
        
    
        
    

'''
simulacao = Simulacao(concentration = 0.35,
                       shape = (100,100,100), 
                       modo_distribuicao = 'fibras',
                       parametros = {'logmean': 0.81,
                                     'logstd': 0.61})

#print(simulacao.simular_percolacao())
'''

def testar_simulacao(n=300):
    start_time = time.process_time()
    simulacao = Simulacao(concentration = 0.35,
                       shape = (n,)*3, 
                       modo_distribuicao = 'fibras',
                       parametros = {'logmean': 0.81,
                                     'logstd': 0.61})
    simulacao.simular_percolacao()
    print('Tempo = ', time.process_time() - start_time)


'''
GUI
'''

'''
def destroy_Toplevel1():
    global w
    w.destroy()
    w = None
'''

class Toplevel1:
    
    
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        #_ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font10 = "-family {Segoe UI} -size 12 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"
        font9 = "-family {Segoe UI} -size 12 -weight bold -slant roman"     \
            " -underline 0 -overstrike 0"
        font8 = "-family {Segoe UI} -size 8 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        self.results = None
        self.top = top
        top.geometry("753x418+338+74")
        top.title("Simulador Discretizado")
        top.configure(background="#d9d9d9")
        
        self.VAR_filler_type = tk.StringVar()
        self.VAR_distribution_type = tk.StringVar()
        self.VAR_filler_mean = tk.StringVar()
        self.VAR_filler_desvpad = tk.StringVar()
        self.VAR_autocalc = tk.StringVar()
        self.VAR_propx = tk.StringVar()
        self.VAR_propy = tk.StringVar()
        self.VAR_propz = tk.StringVar()
        self.VAR_sizes = tk.StringVar()
        self.VAR_concentrations = tk.StringVar()
        self.VAR_repetitions = tk.StringVar()
        
        self.VAR_autocalc.set('0')
        self.VAR_propx.set('1')
        self.VAR_propy.set('1')
        self.VAR_propz.set('1')

        self.VAR_filler_type.trace('w', self.change_filler_type)
        self.VAR_distribution_type.trace('w', self.change_distribution_type)        
                      
        self.Label1 = tk.Label(top)
        self.Label1.place(relx=0.013, rely=0.024, height=27, width=164)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font=font9)
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Parâmetros da Carga''')

        self.Label2 = tk.Label(top)
        self.Label2.place(relx=0.013, rely=0.096, height=27, width=99)
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font=font10)
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(text='''Tipo de carga''')

        self.Label3 = tk.Label(top)
        self.Label3.place(relx=0.013, rely=0.167, height=27, width=89)
        self.Label3.configure(background="#d9d9d9")
        self.Label3.configure(disabledforeground="#a3a3a3")
        self.Label3.configure(font=font10)
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(text='''Distribuição''')

        self.LBL_filler_mean = tk.Label(top)
        self.LBL_filler_mean.place(relx=0.013, rely=0.239, height=27, width=200)
        self.LBL_filler_mean.configure(background="#d9d9d9")
        self.LBL_filler_mean.configure(disabledforeground="#a3a3a3")
        self.LBL_filler_mean.configure(font=font10)
        self.LBL_filler_mean.configure(foreground="#000000")

        self.LBL_filler_desvpad = tk.Label(top)
        self.LBL_filler_desvpad.place(relx=0.013, rely=0.311, height=27
                , width=200)
        self.LBL_filler_desvpad.configure(background="#d9d9d9")
        self.LBL_filler_desvpad.configure(disabledforeground="#a3a3a3")
        self.LBL_filler_desvpad.configure(font=font10)
        self.LBL_filler_desvpad.configure(foreground="#000000")

        self.CBOX_filler_type = ttk.Combobox(top)
        self.CBOX_filler_type.place(relx=0.292, rely=0.096, relheight=0.05
                , relwidth=0.19)
        self.value_list = ['Fibras','Esferas','Pontos',]
        self.CBOX_filler_type.configure(values=self.value_list)
        self.CBOX_filler_type.configure(textvariable=self.VAR_filler_type)
        self.CBOX_filler_type.configure(takefocus="")

        self.CBOX_distribution = ttk.Combobox(top)
        self.CBOX_distribution.place(relx=0.292, rely=0.167, relheight=0.05
                , relwidth=0.19)
        self.value_list = ['Log-Normal','Constante',]
        self.CBOX_distribution.configure(values=self.value_list)
        self.CBOX_distribution.configure(textvariable=self.VAR_distribution_type)
        self.CBOX_distribution.configure(takefocus="")

        self.ENT_filler_mean = tk.Entry(top)
        self.ENT_filler_mean.place(relx=0.292, rely=0.239, height=20
                , relwidth=0.218)
        self.ENT_filler_mean.configure(background="white")
        self.ENT_filler_mean.configure(disabledforeground="#a3a3a3")
        self.ENT_filler_mean.configure(font="TkFixedFont")
        self.ENT_filler_mean.configure(foreground="#000000")
        self.ENT_filler_mean.configure(insertbackground="black")
        self.ENT_filler_mean.configure(textvariable=self.VAR_filler_mean)

        self.ENT_filler_desvpad = tk.Entry(top)
        self.ENT_filler_desvpad.place(relx=0.292, rely=0.287, height=20
                , relwidth=0.218)
        self.ENT_filler_desvpad.configure(background="white")
        self.ENT_filler_desvpad.configure(disabledforeground="#a3a3a3")
        self.ENT_filler_desvpad.configure(font="TkFixedFont")
        self.ENT_filler_desvpad.configure(foreground="#000000")
        self.ENT_filler_desvpad.configure(insertbackground="black")
        self.ENT_filler_desvpad.configure(textvariable=self.VAR_filler_desvpad)

        self.Label4 = tk.Label(top)
        self.Label4.place(relx=0.013, rely=0.383, height=27, width=169)
        self.Label4.configure(background="#d9d9d9")
        self.Label4.configure(disabledforeground="#a3a3a3")
        self.Label4.configure(font=font9)
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(text='''Parâmetros da matriz''')

        self.CBUT_autocalc = tk.Checkbutton(top)
        self.CBUT_autocalc.place(relx=0.279, rely=0.383, relheight=0.06
                , relwidth=0.223)
        self.CBUT_autocalc.configure(activebackground="#ececec")
        self.CBUT_autocalc.configure(activeforeground="#000000")
        self.CBUT_autocalc.configure(background="#d9d9d9")
        self.CBUT_autocalc.configure(command=self.autocalc)
        self.CBUT_autocalc.configure(disabledforeground="#a3a3a3")
        self.CBUT_autocalc.configure(foreground="#000000")
        self.CBUT_autocalc.configure(highlightbackground="#d9d9d9")
        self.CBUT_autocalc.configure(highlightcolor="black")
        self.CBUT_autocalc.configure(justify='left')
        self.CBUT_autocalc.configure(state='active')
        self.CBUT_autocalc.configure(text='''Calcular automaticamente''')
        self.CBUT_autocalc.configure(variable=self.VAR_autocalc)

        self.Label5 = tk.Label(top)
        self.Label5.place(relx=0.013, rely=0.478, height=27, width=81)
        self.Label5.configure(background="#d9d9d9")
        self.Label5.configure(disabledforeground="#a3a3a3")
        self.Label5.configure(font=font10)
        self.Label5.configure(foreground="#000000")
        self.Label5.configure(text='''Proporção:''')

        self.Entry1 = tk.Entry(top)
        self.Entry1.place(relx=0.159, rely=0.478,height=20, relwidth=0.045)
        self.Entry1.configure(background="white")
        self.Entry1.configure(disabledforeground="#a3a3a3")
        self.Entry1.configure(font="TkFixedFont")
        self.Entry1.configure(foreground="#000000")
        self.Entry1.configure(insertbackground="black")
        self.Entry1.configure(textvariable=self.VAR_propx)

        self.Label6 = tk.Label(top)
        self.Label6.place(relx=0.133, rely=0.478, height=27, width=18)
        self.Label6.configure(background="#d9d9d9")
        self.Label6.configure(disabledforeground="#a3a3a3")
        self.Label6.configure(font=font10)
        self.Label6.configure(foreground="#000000")
        self.Label6.configure(text='''X:''')

        self.Label7 = tk.Label(top)
        self.Label7.place(relx=0.212, rely=0.478, height=27, width=18)
        self.Label7.configure(background="#d9d9d9")
        self.Label7.configure(disabledforeground="#a3a3a3")
        self.Label7.configure(font=font10)
        self.Label7.configure(foreground="#000000")
        self.Label7.configure(text='''Y:''')

        self.Entry2 = tk.Entry(top)
        self.Entry2.place(relx=0.239, rely=0.478,height=20, relwidth=0.045)
        self.Entry2.configure(background="white")
        self.Entry2.configure(disabledforeground="#a3a3a3")
        self.Entry2.configure(font="TkFixedFont")
        self.Entry2.configure(foreground="#000000")
        self.Entry2.configure(insertbackground="black")
        self.Entry2.configure(textvariable=self.VAR_propy)

        self.Label8 = tk.Label(top)
        self.Label8.place(relx=0.292, rely=0.478, height=27, width=18)
        self.Label8.configure(background="#d9d9d9")
        self.Label8.configure(disabledforeground="#a3a3a3")
        self.Label8.configure(font=font10)
        self.Label8.configure(foreground="#000000")
        self.Label8.configure(text='''Z:''')

        self.Entry3 = tk.Entry(top)
        self.Entry3.place(relx=0.319, rely=0.478,height=20, relwidth=0.045)
        self.Entry3.configure(background="white")
        self.Entry3.configure(disabledforeground="#a3a3a3")
        self.Entry3.configure(font="TkFixedFont")
        self.Entry3.configure(foreground="#000000")
        self.Entry3.configure(insertbackground="black")
        self.Entry3.configure(textvariable=self.VAR_propz)

        self.Button1 = tk.Button(top)
        self.Button1.place(relx=0.398, rely=0.478, height=24, width=116)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(command=self.reset_cubic)
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Resetar para Cúbico''')

        self.Label9 = tk.Label(top)
        self.Label9.place(relx=0.013, rely=0.598, height=27, width=239)
        self.Label9.configure(background="#d9d9d9")
        self.Label9.configure(disabledforeground="#a3a3a3")
        self.Label9.configure(font=font10)
        self.Label9.configure(foreground="#000000")
        self.Label9.configure(text='''Tamanhos - separar valores com ;''')

        self.Label10 = tk.Label(top)
        self.Label10.place(relx=0.013, rely=0.67, height=27, width=269)
        self.Label10.configure(background="#d9d9d9")
        self.Label10.configure(disabledforeground="#a3a3a3")
        self.Label10.configure(font=font10)
        self.Label10.configure(foreground="#000000")
        self.Label10.configure(text='''Concentrações - separar valores com ;''')

        self.Label11 = tk.Label(top)
        self.Label11.place(relx=0.013, rely=0.766, height=27, width=197)
        self.Label11.configure(background="#d9d9d9")
        self.Label11.configure(disabledforeground="#a3a3a3")
        self.Label11.configure(font=font10)
        self.Label11.configure(foreground="#000000")
        self.Label11.configure(text='''Repetições por combinação''')

        self.ENT_sizes = tk.Entry(top)
        self.ENT_sizes.place(relx=0.385, rely=0.622,height=20, relwidth=0.058)
        self.ENT_sizes.configure(background="white")
        self.ENT_sizes.configure(disabledforeground="#a3a3a3")
        self.ENT_sizes.configure(font="TkFixedFont")
        self.ENT_sizes.configure(foreground="#000000")
        self.ENT_sizes.configure(insertbackground="black")
        self.ENT_sizes.configure(textvariable=self.VAR_sizes)

        self.ENT_concentrations = tk.Entry(top)
        self.ENT_concentrations.place(relx=0.385, rely=0.694, height=20
                , relwidth=0.058)
        self.ENT_concentrations.configure(background="white")
        self.ENT_concentrations.configure(disabledforeground="#a3a3a3")
        self.ENT_concentrations.configure(font="TkFixedFont")
        self.ENT_concentrations.configure(foreground="#000000")
        self.ENT_concentrations.configure(insertbackground="black")
        self.ENT_concentrations.configure(textvariable=self.VAR_concentrations)

        self.ENT_repetitions = tk.Entry(top)
        self.ENT_repetitions.place(relx=0.385, rely=0.766, height=20
                , relwidth=0.058)
        self.ENT_repetitions.configure(background="white")
        self.ENT_repetitions.configure(disabledforeground="#a3a3a3")
        self.ENT_repetitions.configure(font="TkFixedFont")
        self.ENT_repetitions.configure(foreground="#000000")
        self.ENT_repetitions.configure(insertbackground="black")
        self.ENT_repetitions.configure(textvariable=self.VAR_repetitions)

        self.Button2 = tk.Button(top)
        self.Button2.place(relx=0.239, rely=0.837, height=35, width=90)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(command=self.run_simulation)
        self.Button2.configure(disabledforeground="#a3a3a3")
        self.Button2.configure(font=font10)
        self.Button2.configure(foreground="#000000")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(pady="0")
        self.Button2.configure(text='''Simular''')

        '''
        self.CNV_canvas = tk.Canvas(top)
        self.CNV_canvas.place(relx=0.584, rely=0.096, relheight=0.533
                , relwidth=0.376)
        self.CNV_canvas.configure(background="#d9d9d9")
        self.CNV_canvas.configure(borderwidth="2")
        self.CNV_canvas.configure(insertbackground="black")
        self.CNV_canvas.configure(relief="ridge")
        self.CNV_canvas.configure(selectbackground="#c4c4c4")
        self.CNV_canvas.configure(selectforeground="black")
        '''
        self.fig, self.ax = plt.subplots()
        self.CNV_canvas = FigureCanvasTkAgg(self.fig, master = top)
        self.CNV_canvas.get_tk_widget().place(relx=0.584, rely=0.096, relheight=0.533
                , relwidth=0.376)
        self.CNV_canvas.get_tk_widget().configure(background="#d9d9d9")
        self.CNV_canvas.get_tk_widget().configure(borderwidth="2")
        self.CNV_canvas.get_tk_widget().configure(insertbackground="black")
        self.CNV_canvas.get_tk_widget().configure(relief="ridge")
        self.CNV_canvas.get_tk_widget().configure(selectbackground="#c4c4c4")
        self.CNV_canvas.get_tk_widget().configure(selectforeground="black")
        
        self.Button3 = tk.Button(top)
        self.Button3.place(relx=0.584, rely=0.646, height=35, width=111)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(command=self.save_graph)
        self.Button3.configure(disabledforeground="#a3a3a3")
        self.Button3.configure(font=font10)
        self.Button3.configure(foreground="#000000")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(pady="0")
        self.Button3.configure(text='''Salvar Gráfico''')

        self.PROG_progress = ttk.Progressbar(top)
        self.PROG_progress.place(relx=0.013, rely=0.933, relwidth=0.544
                , relheight=0.0, height=22)
        self.PROG_progress.configure(length="410")

        self.Button4 = tk.Button(top)
        self.Button4.place(relx=0.784, rely=0.646, height=35, width=126)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(command=self.export_values)
        self.Button4.configure(disabledforeground="#a3a3a3")
        self.Button4.configure(font=font10)
        self.Button4.configure(foreground="#000000")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(pady="0")
        self.Button4.configure(text='''Exportar valores''')
        
        self.LBL_percolation = tk.Label(top)
        self.LBL_percolation.place(relx=0.474, rely=0.746, height=35, width=350)
        self.LBL_percolation.configure(background="#d9d9d9")
        self.LBL_percolation.configure(disabledforeground="#a3a3a3")
        self.LBL_percolation.configure(font=font8)
        self.LBL_percolation.configure(foreground="#000000")
        
        self.LBL_exponent = tk.Label(top)
        self.LBL_exponent.place(relx=0.474, rely=0.8, height=35, width=350)
        self.LBL_exponent.configure(background="#d9d9d9")
        self.LBL_exponent.configure(disabledforeground="#a3a3a3")
        self.LBL_exponent.configure(font=font8)
        self.LBL_exponent.configure(foreground="#000000")

    def autocalc(self):
        if int(self.VAR_autocalc.get()) == 1:
            self.ENT_sizes.configure(state = tk.DISABLED)
            self.ENT_concentrations.configure(state = tk.DISABLED)
            self.ENT_repetitions.configure(state = tk.DISABLED)
        if int(self.VAR_autocalc.get()) == 0:
            self.ENT_sizes.configure(state = tk.NORMAL)
            self.ENT_concentrations.configure(state = tk.NORMAL)
            self.ENT_repetitions.configure(state = tk.NORMAL)
            
    def export_values(self):
        
        try:
            results = self.results
            current_time = str(datetime.now())[:-10].replace(':','-')
        except NameError:
            pass
        
        with open(current_time+'results.txt',mode = 'w') as file:
            keys = results[0].keys()
            head = '\t'.join(keys)
            file.write(head)
            for res in results:
                file.write('\n' + '\t'.join([str(res[i]) for i in keys]))

    def reset_cubic(self):
        
        self.VAR_propx.set('1')
        self.VAR_propy.set('1')
        self.VAR_propz.set('1')
    
    def run_simulation(self):
        
        variables = {
        'filler_type' : [self.VAR_filler_type.get().lower(), 'string'],
        'distribution_type': [self.VAR_distribution_type.get().lower(), 'string'],
        'filler_mean': [self.VAR_filler_mean.get(), 'float'],
        'filler_desvpad': [self.VAR_filler_desvpad.get().lower(), 'string'],
        'autocalc': [self.VAR_autocalc.get(), 'bool'],
        'propx': [self.VAR_propx.get(), 'float'],
        'propy': [self.VAR_propy.get(), 'float'],
        'propz': [self.VAR_propz.get(), 'float'],
        'sizes': [self.VAR_sizes.get(), 'string'],
        'concentrations': [self.VAR_concentrations.get().lower(), 'string'],
        'repetitions': [self.VAR_repetitions.get().lower(), 'string']
        }
        for var in variables:
            value, kind = variables[var]
            try:
                if kind == 'float': variables[var][0] = float(value)
                elif kind == 'bool': variables[var][0] = bool(int(value))
            except ValueError:
                print(str(var), value, kind)
                return
                
        is_square = (variables['propx'][0] == variables['propy'][0]
                 and variables['propy'][0] == variables['propz'][0])
        if variables['filler_type'] == 'pontos':
            pass
        elif variables['distribution_type'][0] == 'log-normal':
            logmean = variables['filler_mean'][0]
            logstd = float(variables['filler_desvpad'][0])
        elif variables['distribution_type'][0] == 'constante':
            logmean = np.log(variables['filler_mean'][0])
            logstd = 0
        else:
            print(variables['distribution_type'][0], 'not found')
        
        if self.VAR_autocalc.get() == '1':
            use_fast = messagebox.askyesno("Fast", "Use fast datapoints?")
            print(use_fast)
            percolation_bounds = scan_percolation(
                         modo_distribuicao = variables['filler_type'][0].lower(),
                         parametros = {'logmean' : logmean, 'logstd' : logstd})
            expanded_percolation_bounds = [0.8 * percolation_bounds[0],
                                           2.5 * percolation_bounds[1] - percolation_bounds[0]]
            if expanded_percolation_bounds[1] > 1:
                expanded_percolation_bounds[1] = 1
            
            if use_fast:
                datapoints = FAST_AUTO
            else:
                datapoints = SLOW_AUTO
            
            results = []
            
            finished_datapoints = 0
            
            for edge_size, repetitions in datapoints:
                concentrations = np.array(range(repetitions)).astype('float64')
                concentrations *= (expanded_percolation_bounds[1] 
                                 - expanded_percolation_bounds[0]) / repetitions
                concentrations += expanded_percolation_bounds[0]
                
                finished_concentrations = 0
                
                for concentration in concentrations:
                    
                    shape = (int(variables['propx'][0] * edge_size),
                             int(variables['propy'][0] * edge_size),
                             int(variables['propz'][0] * edge_size))
                    
                    sim = Simulacao(concentration = concentration,
                                    shape = shape,
                                    modo_distribuicao = variables['filler_type'][0].lower(),
                                    parametros = {'logmean' : logmean, 'logstd' : logstd})
                    percolations, interconectivity = sim.simular_percolacao()
                    if is_square:
                        results.append( {'size' : edge_size,
                                              'concentration' : concentration,
                                              'percolations' : sum(percolations)/3,
                                              'interconectivity' : sum(interconectivity)/3})
                    else:
                        results.append({ 'size' : edge_size, 
                                              'concentration' : concentration,
                                              'percolations' : percolations[2],
                                              'interconectivity' : interconectivity[2]})
                    finished_concentrations += 1
                    progress_value = 100 * (finished_datapoints + finished_concentrations / len(concentrations)) / len(datapoints)
                    print(progress_value)
                    self.PROG_progress.configure(value=progress_value)
                    self.top.update()
                finished_datapoints += 1
                
        if self.VAR_autocalc.get() == '0':
            
            results = []
            finished_datapoints = 0
            
            try:
                repetitions = int(variables['repetitions'][0])
            except ValueError:
                print('Repetitions is invalid')
                return
            
            try:
                sizes = [int(i.strip()) for i in variables['sizes'][0].split(';')]
                concentrations = [float(i.strip()) for i in variables['concentrations'][0].split(';')]
            except ValueError:
                print('Sizes and/or concentrations is invalid')
                return
            
            datapoints = [ (i,j) for i in sizes for j in concentrations]
            
            for edge_size, concentration in datapoints:
                
                finished_concentrations = 0
                
                for _ in range(repetitions):
                    
                    shape = (int(variables['propx'][0] * edge_size),
                             int(variables['propy'][0] * edge_size),
                             int(variables['propz'][0] * edge_size))
                    
                    sim = Simulacao(concentration = concentration,
                                    shape = shape,
                                    modo_distribuicao = variables['filler_type'][0].lower(),
                                    parametros = {'logmean' : logmean, 'logstd' : logstd})
                    percolations, interconectivity = sim.simular_percolacao()
                    if is_square:
                        results.append( {'size' : edge_size,
                                              'concentration' : concentration,
                                              'percolations' : sum(percolations)/3,
                                              'interconectivity' : sum(interconectivity)/3})
                    else:
                        results.append({'size' : edge_size,  
                                              'concentration' : concentration,
                                              'percolations' : percolations[2],
                                              'interconectivity' : interconectivity[2]})
                    finished_concentrations += 1
                    progress_value = 100 * (finished_datapoints + finished_concentrations / len(concentrations)) / len(datapoints)
                    print(progress_value)
                    self.PROG_progress.configure(value=progress_value)
                    self.top.update()
                finished_datapoints += 1
                
        self.results = results
        #print(results)
        
        sizes = np.unique([i['size'] for i in results])
        
        scatter_points = []
        
        for size in sizes:
            concentrations = [i['concentration'] for i in results if i['size'] == size]
            interconectivity = [i['interconectivity'] for i in results if i['size'] == size]
            scatter_points.append((concentrations, interconectivity))
            
        self.scatter_points = scatter_points
        
        #fig, ax = plt.subplots()
        self.ax.set_ylabel('Interconectivity')
        self.ax.set_xlabel('Concentration')
        plt.tight_layout()
        for scatter in scatter_points:
            #print(scatter[0], scatter[1])
            self.ax.scatter(scatter[0], scatter[1])
            
        #plt.show()
        self.CNV_canvas.draw()
        
        fits = []
        bounds = ([THRESHOLD_BOUNDS[0], EXPONENT_BOUNDS[0]],
                  [THRESHOLD_BOUNDS[1], EXPONENT_BOUNDS[1]])
        
        for scatter in scatter_points:
            
            under_threshold = [i for i in scatter if i[1] == 0]
            over_threshold = [i for i in scatter if i[1] > 0]
            
            if len(under_threshold) == 0 or len(over_threshold) == 0:
                threshold_0 = 0.5
            else:
                threshold_0 = (max([i[0] for i in under_threshold])
                             + min([i[0] for i in over_threshold])) / 2
            
            exponent_0 = [np.log(i[1]) / (np.log(i[0]) - threshold_0) 
                           for i in over_threshold]
            exponent_0 = np.array(exponent_0).mean()
            
            popt, pcov = optimize.curve_fit(percolation_function,
                                            scatter[0],
                                            scatter[1],
                                            p0 = [threshold_0, exponent_0],
                                            bounds = bounds)
            
            fits.append((popt, pcov))
        
        threshold = None
        threshold_ci = np.inf
        exponent = None
        exponent_ci = np.inf
        for fit in fits:
            fit_threshold, fit_exponent = fit[0]
            fit_threshold_ci, fit_exponent_ci = 1.96 * np.sqrt(np.diag(pcov))
            if fit_threshold_ci < threshold_ci:
                threshold_ci = fit_threshold_ci
                exponent_ci = fit_exponent_ci
                threshold = fit_threshold
                exponent = fit_exponent
        
        self.LBL_percolation.configure(text = f'Limiar de percolação: {threshold:.6f} +- {threshold_ci:.6f}')
        self.LBL_exponent.configure(text = f'Coeficiente: {exponent:.6f} +- {exponent_ci:.6f}')
        
        
        
    def save_graph(self):
        print('Saving graphs')
        current_time = str(datetime.now())[:-10].replace(':','-')
        self.fig.savefig(current_time + '_graph.png')
        
        try:
            current_time = str(datetime.now())[:-10]
            self.fig.savefig(current_time + '_graph.png')
        except NameError:
            pass
        
    def change_filler_type(self, *args):
        
        if self.VAR_filler_type.get() == 'Disperso':
            self.CBOX_distribution.configure(state = tk.DISABLED)
            self.ENT_filler_mean.configure(state = tk.DISABLED)
            self.ENT_filler_desvpad.configure(state = tk.DISABLED)
            
        else:
            self.CBOX_distribution.configure(state = tk.NORMAL)
            self.ENT_filler_mean.configure(state = tk.NORMAL)
            self.ENT_filler_desvpad.configure(state = tk.NORMAL)

    def change_distribution_type(self, *args):
        print(self.VAR_distribution_type.get())
        if self.VAR_distribution_type.get() == 'Log-Normal':
            print('Log-Normal')
            self.ENT_filler_mean.configure(state = tk.NORMAL)
            self.ENT_filler_desvpad.configure(state = tk.NORMAL)
            self.LBL_filler_mean.configure(text = 'Média Logarítmica: ')
            self.LBL_filler_desvpad.configure(text = 'Desvio padrão: ')
            
        elif self.VAR_distribution_type.get() == 'Constante':
            self.ENT_filler_mean.configure(state = tk.NORMAL)
            self.ENT_filler_desvpad.configure(state = tk.DISABLED)
            self.LBL_filler_mean.configure(text = 'Valor constante: ')
            self.LBL_filler_desvpad.configure(text = ' ')

    
if START_GUI == True:
    root = tk.Tk()
    top = Toplevel1(root)
    root.mainloop()



