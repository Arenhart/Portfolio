'''
Created on 21/11/2012

@author: Rafael

'''

from __future__ import division
import random as rd
from numpy import random, sin, cos, arccos, dot, cross, std
from math import pi, sqrt, log
import time
import math
import os
import itertools

random.random()
time.clock()


# text inputs with multi-language support
PARTICLE_SHAPES = [["sphere","fiber","cylinder"],
                   ["esfera","esfera","cilindro"]]
SIZE_DISTRIBUTIONS = [["constant","normal","lognormal"],
                      ["constante","normal","lognormal"]]
INPUT_PROMPT = [["Particle shape (", "Size distribution (", "Distribution parameters ("],
                 ["Formato da particula (", "Distribuicao de tamanho (", "parametros da distribuicao ("]]
DISTRIBUTION_PROMPT = [{"constant" : ["value"] , "normal": ["mean", "standard deviation"], "lognormal":["location","scale", 'maximum']},
                       {"constant" : ["valor"] , "normal": ["media", "desvio padrao"], "lognormal":["localizacao","scala", 'maximo']}]
GEOMETRY_PROMPT = {"sphere" : [["radius"],
                               ["raio"]],
                   "fiber" : [["radius","length"],
                              ["raio","comprimento"]],
                   "cylinder" : [["radius", "length"],
                                 ["raio","comprimento"]]}
OTHER_MESSAGES = [["Not a number", "Particle frequency (std=1): ","Add another particle? (", "y", "n", "Is the particle conductive? (","y","n"],
                  ["Valor nao-numerico", "Frequencia da particula (pad=1): ","adicionar outra particula? (", "s","n","A particula e condutora? (","s","n"]]
LANGUAGES = {"English" : 0, "Portugues" : 1}


# constants for the model operation

FRACTION_TOLERANCE = 0.02 

SECTORIZATION_MODEL = 1 #only one available

SOFT_SHELL = 0.01 #ratio of particle radius that can interpenetrate

DEBUG = 0 # overruns the input phase and runs a standard particle setup

VERBOSE_DEBUG = 0 #runs several prints to describe process steps

OUTPUT = 0 # generates files to be run in blender to generate 3D images, both of initial softcore shape, intermediate 

HARDCORE = 1 # if 1, will run the accommodation code to simulate hardcore spheres with soft shell

HARDCORE_TOLERANCE = 20 # after how many iterations the hardcore model approximatelly doubles the softcore value

HARDCORE_INTERRUPT = 500 # after how many iterations the hardcore model is interrupted

Z_TOLERANCE = None # absolute distance from the XY plane (either top or bottom) that particles will still be considered in contact with for conductive path formation. 
                #(values higher that zero mean the particle does not need to touch the plane to be in contact with it)

CYLINDER_AS_DISK = 1 # represent all cylinders as disks, currently must be always 1

ACCOMMODATION_STEP = 0.1 # must be lower than 1. Multiplies the displacement vector when accommodating particles, to avoid displacing beyond the contact range

MAX_ANGLE_X, MAX_ANGLE_Z, MIN_ANGLE_Z = 1, 1, 0 #defines alignment of fibers
#examples:
# 1, 1, 0: Fibers unaligned
# 0.1, 1, 0: Fibers on the ZY plane
# 1, 0.1, 0: Fibers perpendicular to the Z axis
# 1, 1, 0.9: Fibers aligned with the Z axis

SIMPLE_BLENDER_FIBER = 1 #Removes the caps on the blender output file

CALCULATE_CONDUCTIVITY = 0

#global variables
particle_templates = [] # a list to be filled with particle template objects. They handle addition of particles
                        # to the simulation volume and contacts between them
language = "English"
lang_index = LANGUAGES[language] #get the index of the selected language
ini = time.clock()
file_number = 0

# helper functions

def roll_distance_normal(avg,std):
    dist = rd.gauss(avg,std)
    if dist <= 0.001: dist = 0.001
    return dist

def roll_distance_flat(low,high):
    dist = rd.random()*(high-low)+low
    return dist

def S(s,sigma_cla,tunnel_a,tunnel_b,a):
    '''
    calculates the conductivity based on the approximation of Simmons model
    s is distance in angstroms, a is cross area in square angstroms, 
    the units of the conductivities depend on the situation
    '''
    s_tun = 10**(tunnel_a-tunnel_b*s)*a
    s_cla = (sigma_cla*a)/(s*10**-8)
    if s_tun >= s_cla: return s_tun
    else: return s_cla
    

#def sqrt(x):
#    return x**0.5

def SD1(x_i,xp,yp,a,b):
    #first derivative of squared distance
    y = b*math.sqrt(1-(x_i/a)**2)
    return x_i*((-2*b**2)/(a**2)+(2*y*yp)/(a**2-x_i**2)+2)-2*xp

def SD2(x_i,xp,yp,a,b):
    #second derivative of squared distance
    y = b*math.sqrt(1-(x_i/a)**2)
    return (2*(a**4 + a**2 * (b*y-2*x_i**2) +x_i**4)) / (a**2-x_i**2)**2 - (2*b**2)/a**2

def find_closest_point_in_ellipse(x,y,a,b):
    
    x, y = abs(x), abs(y)
    x_i = 0
    count = 0
    while 1:
        delta_x = SD1(x_i,x,y,a,b)/SD2(x_i,x,y,a,b)
        while abs(x_i) < abs(x_i-delta_x): delta_x = delta_x / 2
        new_x = x_i - delta_x
        if math.fabs(x_i - new_x) <= 0.00000001 : break
        x_i = new_x
        if x_i >= a: 
            x_i = a*0.9999999
        count += 1
        if count > 10000: 
            print (x,y,a,b)
            count = 0
        
    y_i = b*math.sqrt(1-(x_i/a)**2)
    return (x_i, y_i)

def find_distance_to_ellipse(x,y,a,b):
    # finds shortest distance from a point (x,y) to the ellipse with major axis a and minor axis b
    if x == 0:
        return math.fabs(y-b)
    closest_point = find_closest_point_in_ellipse(x,y,a,b)
    x_p, y_p = closest_point[0], closest_point[1]
    return math.sqrt((x_p - x)**2 + (y_p - y) **2)

def blender_file(number):
    '''
    creates a text file with a python script that can be run in blender to create spheres representing the particles
    currently, spherocylinders are represented as spheres
    '''
    
    directory = os.getcwd()+'\\blender output'
    try:
        f = open(directory +'\\blender_output_'+str(number)+'.txt','w')
    except:
        os.mkdir(directory)
        f = open(directory +'\\blender_output_'+str(number)+'.txt','w')
    
    color_dict = {}
    
    for particle in simulation.particle_library :
                 
        
        if str(simulation.root(particle)) not in color_dict:
                color_dict[str(simulation.root(particle))] = len(color_dict)+1
                particle['color'] = len(color_dict)               
        else: 
            particle['color'] = color_dict[str(simulation.root(particle))]
           
    
    f.write('MAX_ROOT = ' + str(len(color_dict)+1) + '\n')
    for particle in simulation.particle_library:

        if particle['shape'] == 'sphere':
            if particle['conductive'] == 1:
                f.write('add_sphere(' + str(particle['center']) + ', ' +  str(particle['geometric parameters']['radius']) + ', '  + str(particle['color']) +')\n')
            elif particle['conductive'] == 0:
                f.write('add_sphere(' + str(particle['center']) + ', ' +  str(particle['geometric parameters']['radius']) + ', '  + '0' +')\n')
        
        elif particle['shape'] == 'fiber':
            if particle['conductive'] == 1:
                f.write('add_cylinder(' + 
                        str(particle['center']) + ', ' +
                        str(particle['geometric parameters']['length']) + ', '  + 
                        str(particle['geometric parameters']['radius']) + ', '  +
                        str(particle['geometric parameters']['rotation']) + ', '  +  
                        str(particle['color']) +')\n')
                if not SIMPLE_BLENDER_FIBER:
                    front_point = [j+k for (j,k) in zip(particle['center'],particle['geometric parameters']['half shaft'])]
                    f.write('add_sphere(' + str(front_point) + ', ' +  str(particle['geometric parameters']['radius']) + ', '  + str(particle['color']) +')\n')
                    back_point = [j-k for (j,k) in zip(particle['center'],particle['geometric parameters']['half shaft'])]
                    f.write('add_sphere(' + str(back_point) + ', ' +  str(particle['geometric parameters']['radius']) + ', '  + str(particle['color']) +')\n')
                
            elif particle['conductive'] == 0:
                f.write('add_cylinder(' + 
                        str(particle['center']) + ', ' +
                        str(particle['geometric parameters']['length']) + ', '  + 
                        str(particle['geometric parameters']['radius']) + ', '  +
                        str(particle['geometric parameters']['rotation']) + ', '  +  
                        '0' +')\n')
    f.close()

def create_sphere(center,r):
    '''
    creates the output format for an octaedra for blender output
    currently unused
    '''
    vertexes = []
    faces = []
    x=center[0]
    y=center[1]
    z=center[2]
    #top vertex
    vertexes.append((x,y,z+r))
    
    #center vertexes
    vertexes.extend(((x+r,y+r,z),(x+r,y-r,z),(x-r,y-r,z),(x-r,y+r,z)))
    
    #bottom vertex
    vertexes.append((x,y,z-r))
    
    vertexes = tuple(vertexes)
    
    #top faces
    faces.extend(((0,1,2),(0,2,3),(0,3,4),(0,4,1)))
    
    #bottom faces
    faces.extend(((5,1,2),(5,2,3),(5,3,4),(5,4,1)))
    
    faces = tuple(faces)
    
    return str(vertexes)

def calc_contact_vector(particle1,particle2):
    '''
    returns the normalized vector of contact between particle1 and particle2
    '''
    
    if particle1['shape'] == 'sphere':
        if particle2['shape'] == 'sphere' or particle2['shape'] == 'disk':
            contact_vector = [i1-i2 for i1,i2 in zip(particle1['center'],particle2['center'])]
        
        elif particle2['shape'] == 'fiber':
            contact_vector = sphere_fiber_distance(particle1, particle2, SOFT_SHELL, return_vector=True)
    
    elif particle1['shape'] == 'fiber':
        if particle2['shape'] == 'sphere':
            contact_vector = sphere_fiber_distance(particle2, particle1, SOFT_SHELL, return_vector=True)
            
        elif particle2['shape'] == 'fiber':
            contact_vector = fiber_fiber_distance(particle1, particle2, SOFT_SHELL, return_vector=True)
            
        elif particle2['shape'] == 'disk':
            contact_vector = [i1-i2 for i1,i2 in zip(particle1['center'],particle2['center'])]
            
    elif particle1['shape'] == 'disk':
        
        contact_vector = [i1-i2 for i1,i2 in zip(particle1['center'],particle2['center'])]
    
    
    
    vector_length = calc_vector_length(contact_vector)
    normalized_contact_vector = [abs(i/vector_length) for i in contact_vector]        
    return normalized_contact_vector
            

def vector_subtract(vector1,vector2):
    '''
    subtracts one 3 component vector from another 3 component vector
    '''
    return(vector1[0]-vector2[0],vector1[1]-vector2[1],vector1[2]-vector2[2])

def per(vector1, vector2):
    '''
    returns a unit vector perpendicular to input vectors
    '''
    cross_product = cross(vector1,vector2)
    return cross_product / sqrt(sum([i**2 for i in cross_product]))

def indexed(l):
    return zip(range(len(l)),l)    
    

def check_contact(particle1, particle2, soft_shell = 0, override_conductivity = 0):
    # in the test computer, the code performs one million contact checks in 3 seconds (sphere-sphere)
    
    
    # if both particles are in the same aggregate and conductivity is in question, return false
    if particle1['root'] == particle2['root'] and override_conductivity == 0: return 0
    
    # if not both particles are conductive, returns false
    if particle1['conductive'] + particle2['conductive'] != 2 and override_conductivity == 0: return 0

    contact = 0
    if particle1['shape'] == 'sphere':
        if particle2['shape'] == 'sphere':
            contact = sphere_sphere_contact(particle1,particle2,soft_shell)
        elif particle2['shape'] == 'fiber':
            contact = sphere_fiber_contact(particle1,particle2,soft_shell)
        elif particle2['shape'] == 'disk':
            contact = sphere_disk_contact(particle1,particle2,soft_shell)
            
    elif particle1['shape'] == 'fiber':
        if particle2['shape'] == 'sphere':
            contact = sphere_fiber_contact(particle2,particle1,soft_shell)
        elif particle2['shape'] == 'fiber':
            contact = fiber_fiber_contact(particle1,particle2,soft_shell)
        elif particle2['shape'] == 'disk':
            contact = fiber_disk_contact(particle1,particle2,soft_shell)
            
    elif particle1['shape'] == 'disk':
        if particle2['shape'] == 'sphere':
            contact = sphere_disk_contact(particle2,particle1,soft_shell)
        elif particle2['shape'] == 'fiber':
            contact = fiber_disk_contact(particle2,particle1,soft_shell)
        elif particle2['shape'] == 'disk':
            contact = disk_disk_contact(particle1,particle2,soft_shell)
        
    else: print('method not found')
    
    
    return contact

def check_contact_plane(particle, plane, top = False):
    if particle['shape'] == 'sphere':
        if plane == 'z':
            return sphere_zplane_contact(particle,top)
    elif particle['shape'] == 'fiber':
        if plane == 'z':
            return fiber_zplane_contact(particle,top)
    elif particle['shape'] == 'disk':
        if plane == 'z':
            return disk_zplane_contact(particle,top)

def sphere_zplane_contact(particle,top):
    
    if top:
        return particle['center'][2] + particle['geometric parameters']['radius'] >= simulation.len_z - Z_TOLERANCE
    elif not top: 
        return particle['center'][2] - particle['geometric parameters']['radius'] <= Z_TOLERANCE


def sphere_sphere_contact(particle1,particle2,soft_shell):
    

    sqr_distance = calc_squared_distance_3d(particle1['center'], particle2['center'])
    try:
        sqr_sum_of_radius = ((particle1['geometric parameters']['radius']+particle2['geometric parameters']['radius'])*(1-soft_shell))**2
    except:
        print(particle1['geometric parameters']['radius'], 
              particle2['geometric parameters']['radius'], 1-soft_shell)
        0/0
        
    if sqr_distance <= sqr_sum_of_radius:
        return 1
    else:
        return 0

def fiber_fiber_contact(particle1, particle2, soft_shell, return_vector = False):
        
    x1 = particle1['center']
    a1 = particle1['geometric parameters']['half shaft']
    x2 = particle2['center']
    a2 = particle2['geometric parameters']['half shaft']
    r1 = particle1['geometric parameters']['radius']
    r2 = particle2['geometric parameters']['radius']

    #define constraining boxes, first coordinates of positive vertex, then of negative vertex
    box1={}
    box1['+'] = ([i + abs(j) + r1 for i,j in zip(x1,a1)])
    box1['-'] = ([i - abs(j) - r1 for i,j in zip(x1,a1)])
     
    box2={}
    box2['+'] = ([i + abs(j) + r2 for i,j in zip(x2,a2)])
    box2['-'] = ([i - abs(j) - r2 for i,j in zip(x2,a2)])
     
    # makes preliminary check of box collision
    if box1['+'][0] < box2['-'][0] : return False
    if box1['-'][0] > box2['+'][0] : return False
    if box1['+'][1] < box2['-'][1] : return False
    if box1['-'][1] > box2['+'][1] : return False
    if box1['+'][2] < box2['-'][2] : return False
    if box1['-'][2] > box2['+'][2] : return False
    
    if not return_vector:
        distance = fiber_fiber_distance(particle1, particle2, soft_shell, return_vector = False)
        
        if distance <= (particle1['geometric parameters']['radius'] + particle2['geometric parameters']['radius'])*(1-soft_shell): 
            return True        
        else: 
            return False
        
    else: #return_Vector == True
        
        vector = fiber_fiber_distance(particle1, particle2, soft_shell, return_vector)
        distance = (dot(vector,vector))**(1/2)
        interpenetrating_distance = particle1['geometric parameters']['radius'] + particle2['geometric parameters']['radius'] - distance
        displacement_vector = [i*(interpenetrating_distance/distance) for i in vector]
        
        if distance <= (particle1['geometric parameters']['radius'] + particle2['geometric parameters']['radius'])*(1-soft_shell): 
            return displacement_vector        
        else: 
            return False

def fiber_fiber_distance(particle1, particle2, soft_shell, return_vector = False):
    x1 = particle1['center']
    a1 = particle1['geometric parameters']['half shaft']
    x2 = particle2['center']
    a2 = particle2['geometric parameters']['half shaft']
    
    #if calc_squared_distance_3d(x1,x2) > (r1 + r2 + particle1[2]['length']/2 + particle2[2]['length']/2)**2: return False
    
    A = dot(a1,a1)
    B = - dot(a1,a2)
    C = dot(a2,a2)
    D = dot(a1, (x1[0]-x2[0], x1[1]-x2[1], x1[2]-x2[2]) )
    E = - dot(a2, (x1[0]-x2[0], x1[1]-x2[1], x1[2]-x2[2]) )
    delta = A*C-B**2
    
    '''
    due to computational limits, if delta is too low, it cannot calculate delta
    in case delta is too low, the algorithm approximates the particle to two parallel cylinders
    '''
    if delta < 0.00000000001:

        t2 = 0
        t1 = t2 * (D/A)
        t1 = alfa_func(t1)
        t2_scale = sqrt(dot(a1,a1)/dot(a2,a2))
        t2 = t1 - D/A
        t2 = t2_scale * alfa_func(t2/t2_scale)
        t1 = t2 - D/A    

    else:

        t2 = alfa_func((B*D-A*E)/delta)
        temp = (-B*t2-D)/A
        t1 = alfa_func(temp)
        if temp < -1 or temp > 1:
            t2 = alfa_func((-B*t1-E)/C)
        
    point1 = (x1[0]+t1*a1[0], x1[1]+t1*a1[1], x1[2]+t1*a1[2])
    point2 = (x2[0]+t2*a2[0], x2[1]+t2*a2[1], x2[2]+t2*a2[2])
    difference = (point2[0] - point1[0], point2[1]-point1[1], point2[2]-point1[2])
    
    if return_vector:

        return difference
    
    else: #return_vector==False
        return (dot(difference,difference))**(1/2)
  
  
def sphere_fiber_contact(sphere, fiber, soft_shell, return_vector = False):
    
    if calc_distance_3d(sphere['center'],fiber['center']) > sphere['geometric parameters']['radius'] + fiber['geometric parameters']['radius'] + fiber['geometric parameters']['length']/2:
        return False
    
    if not return_vector:
        distance = sphere_fiber_distance(sphere, fiber, soft_shell, return_vector)
        
        if distance <= (fiber['geometric parameters']['radius'] + sphere['geometric parameters']['radius'])*(1-soft_shell): 
            return True
        else:
            return False  
        
    else: #return_vector == True:
        
        vector = sphere_fiber_distance(sphere, fiber, soft_shell, return_vector)
        distance = (dot(vector,vector))**(1/2)
        interpenetrating_distance = sphere['geometric parameters']['radius'] + fiber['geometric parameters']['radius'] - distance
        displacement_vector = [i*(interpenetrating_distance/distance) for i in vector]
        
        if distance <= (fiber['geometric parameters']['radius'] + sphere['geometric parameters']['radius'])*(1-soft_shell): 
            return displacement_vector
        else:
            return False 


def sphere_fiber_distance(sphere, fiber, soft_shell, return_vector = False):
    
    x1 = fiber['center']
    a = fiber['geometric parameters']['half shaft']
    x2 = sphere['center']
    
    t = dot(vector_subtract(x1,x2),vector_subtract(a,x1))/dot(vector_subtract(a,x1),vector_subtract(a,x1))
    
    t = alfa_func(t)
    
    point = (x1[0]+t*a[0], x1[1]+t*a[1], x1[2]+t*a[2])
    
    difference = vector_subtract(point,x2)
    
    if not return_vector:
        return dot(difference,difference)**(1/2)
    
    else:
        return difference
    
    
def fiber_zplane_contact(particle, top):
    
    if top:
        if particle['center'][2] + particle['geometric parameters']['radius'] + abs(particle['geometric parameters']['half shaft'][2]) >= simulation.len_z - Z_TOLERANCE:
            return True
        else:
            return False
    elif not top: 
        if particle['center'][2] - particle['geometric parameters']['radius'] - abs(particle['geometric parameters']['half shaft'][2]) <= Z_TOLERANCE:
            return True
        else:
            return False

def disk_disk_contact(particle1, particle2, soft_shell):
    
    
    p1 = particle1['center']
    p2 = particle2['center']
    a1 = particle1['geometric parameters']['normal vector']
    a2 = particle2['geometric parameters']['normal vector']
    r1 = particle1['geometric parameters']['radius']
    r2 = particle2['geometric parameters']['radius']
    
    
    
    A1, B1, C1 = a1[0], a1[1], a1[2]
    A2, B2, C2 = a2[0], a2[1], a2[2]
    alfa1, beta1, gama1 = p1[0], p1[1], p1[2]
    alfa2, beta2, gama2 = p2[0], p2[1], p2[2]
    
    scalar3 = sqrt( 1 - (A1*A2 + B1*B2 + C1*C2)**2)
    if scalar3 ==0: return False
    A3 = (B1*C2-C1*B2)/scalar3
    B3 = (C1*A2-A1*C2)/scalar3
    C3 = (A1*B2-B1*A2)/scalar3
    
    scalar4 = sqrt( 1 - (A1*A3 + B1*B3 + C1*C3)**2)
    A4 = (B1*C3-C1*B3)/scalar4
    B4 = (C1*A3-A1*C3)/scalar4
    C4 = (A1*B3-B1*A3)/scalar4 
    
    scalar5 = sqrt( 1 - (A2*A3 + B2*B3 + C2*C3)**2)
    A5 = (B2*C3-C2*B3)/scalar5
    B5 = (C2*A3-A2*C3)/scalar5
    C5 = (A2*B3-B2*A3)/scalar5
    
    Dalfa, Dbeta, Dgama = alfa2-alfa1, beta2-beta1, gama2-gama1
    K1 = B5*C3 + B3*C5
    K2 = A3*C5 - A5*C3
    K3 = A3*B5 - B3*A5
    
    denominator1 = ((-A4)*K1+B4*K2+C4*K3)
    if denominator1 == 0: 
        print("!")
        return False
    t1 = ((-Dalfa)*K1+Dbeta*K2+Dgama*K3) / denominator1
    if abs(t1) > r1: return False
    
    denominator2 = (A3*B5 - A5*B3)
    if denominator2 == 0: 
        return False
    t2 = (Dalfa*B3 - t1*A4*B3 - Dbeta*A3 + t1*A3*B4) / denominator2
    if abs(t2) > r2: return False
    
    if C3 != 0: 
        t3 = (Dgama - C4*t1 + C5*t2) / C3
    elif B3 != 0:
        t3 = (Dbeta-t1*B4+t2*B5)/B3
    elif A3 != 0:
        t3 = (Dalfa-t1*A4+t2*A5)/A3
    else:
        return False
    if abs(t3) > sqrt(r1**2-(r1-abs(t1))**2) + sqrt(r2**2-(r2-abs(t2))**2): return False
    
    return True

def sphere_disk_contact(particle1, particle2, soft_shell):
    p1 = particle1['center']
    p2 = particle2['center']
    a2 = particle2['geometric parameters']['normal vector']
    r1 = particle1['geometric parameters']['radius']
    r2 = particle2['geometric parameters']['radius']
    A2, B2, C2 = a2[0], a2[1], a2[2]
    alfa1, beta1, gama1 = p1[0], p1[1], p1[2]
    alfa2, beta2, gama2 = p2[0], p2[1], p2[2]
    
    d = - (A2*alfa2 + B2*beta2 + C2*gama2)
    
    D = (A2*alfa1 + B2*beta1 + C2*gama1 + d) / sqrt(A2**2 + B2**2 + C2**2)
    
    if D > r1: return False
    
    P = (alfa2-D*A2, beta2 - D*B2, gama2 - D*C2)

    if sqrt((P[0]-p1[0])**2+(P[1]-p1[1])**2+(P[2]-p1[2])**2) <= r1 + sqrt(r2**2 - D**2):
        return True
    return False


def fiber_disk_contact(particle1, particle2, soft_shell):
    
    if particle1['shape'] != 'fiber' or particle2['shape'] != 'disk':
        print('geometries dont match (fiber & disk)')
    
    fiber = particle1 # defines the fiber particle
    disk = particle2 # defines the disk particle
    
    # Characteristics of the fiber    
    pe = fiber['center'] # the fibers center of mass
    ae = fiber['geometric parameters']['half shaft'] # the fibers actual half-shaft
    aeu = ae / (dot(ae,ae))**0.5 # the fibers unit half-shaft
    re = fiber['geometric parameters']['radius']
    
    # characteristics of the disk
    pd = disk['center'] # the disks center of mass
    adu = disk['geometric parameters']['normal vector'] # the disks normal vector
    rd = disk['geometric parameters']['radius']
    
    # characteristics of the projection
    # auxiliary characteristics
    D_pe_pld = dot(adu,[i-j for i,j in zip(pe,pd)]) # Shortest distance between center of the fiber and the plane of the disk
    sec_theta = 1 / dot(adu,aeu) # sec theta of vectors adu and aeu
    p1 = [i+j for i,j in zip(pe,ae)] # center point of fiber half cap on positive half-shaft extreme
    p2 = [i-j for i,j in zip(pe,ae)] # center point of fiber half cap on negative half-shaft extreme
    D_p1_pld = dot(adu,[i-j for i,j in zip(p1,pd)])
    D_p2_pld = dot(adu,[i-j for i,j in zip(p2,pd)])
    if D_p1_pld < D_p2_pld:
        p1, p2 = p2,p1
        D_p1_pld, D_p2_pld = D_p2_pld, D_p1_pld
    v = per(per(adu,aeu),aeu) # unit vector of shortest distance between p1 and the disk plane that is also perpendicular to the fiber axis, always points to the opposite direction of adu
    v1 = v * (D_p1_pld/dot(v,adu))
    v2 = v * (D_p2_pld/dot(v,adu))
    mod_v1 = sqrt(dot(v1,v1))
    mod_v2 = sqrt(dot(v2,v2))
    
    # calculate cp
    if   D_pe_pld >= 0:
        if   mod_v1 >= re : cp = 1
        elif mod_v1 < re : cp = mod_v1/re
    elif D_pe_pld < 0:
        if   D_p1_pld >= 0:
            if   mod_v1 >= re: cp = 1
            elif mod_v1 < re: cp = mod_v1/re
        elif D_p1_pld < 0:
            if   mod_v1 >= re: return False
            elif mod_v1 < re: cp = -mod_v1/re
            
    # calculate cn
    if   D_pe_pld < 0:
        if   mod_v2 >= re : cn = 1
        elif mod_v2 < re : cn = mod_v2/re
    elif D_pe_pld >= 0:
        if   D_p2_pld <= 0:
            if   mod_v2 >= re: cn = 1
            elif mod_v2 < re: cn = mod_v2/re
        elif D_p2_pld > 0:
            if   mod_v2 >= re: return False
            elif mod_v2 < re: cn = -mod_v2/re
    
    
    # effective characteristics
    pp = [i - j * D_pe_pld * sec_theta for i,j in zip(pe,aeu)] # center of the projection
    bp = re # smaller axis of projection
    ap = abs(re * sec_theta) # larger axis, disregarding cuts
    
    v3 = [i + j for i,j in zip(p1,v1)]
    v3 = [i - j for i,j in zip(v3,pp)] #defines a vector that is a projection of the half-axis in the disk plane
    
    pd_minus_pp = [i-j for i,j in zip(pd,pp)]
    cos_psi = dot(v3,pd_minus_pp)
    cos_psi = cos_psi / sqrt(dot(v3,v3)*dot(pd_minus_pp,pd_minus_pp)) #psi is the angle between the larger projection axis and v3
    sin_psi = sqrt(1-cos_psi**2)
    
    mod_pd_minus_pp = sqrt(dot(pd_minus_pp,pd_minus_pp))
    
    pd_2d = (mod_pd_minus_pp*cos_psi, mod_pd_minus_pp*sin_psi)
    
    xp = ap*cp
    xn = ap*cn
    
    bpos = sqrt( (1-(xp**2/ap**2)) / bp**2 )
    bneg = sqrt( (1-(xn**2/ap**2)) / bp**2 )
    dp_in_ellipse = (pd_2d[0]/ap)**2 + (pd_2d[1]/bp)**2 <= 1
    
    #contact conditions check:
    # A and B
    if pd_2d[0] >= xn and pd_2d[0] <= xp: 
        if dp_in_ellipse: return True
        elif find_distance_to_ellipse(pd_2d[0],pd_2d[1],ap,bp) <= rd: return True
        else: return False
    # C
    elif pd_2d[0] < xn:
        if calc_squared_distance_2d(pd_2d,(xn,0)) <= (bneg+rd)**2 : return True
        else: return False
    # D
    else: # if pd_2d[0] > xp
        if calc_squared_distance_2d(pd_2d,(xp,0)) <= (bpos+rd)**2 : return True
        else: return False
        
    

def disk_zplane_contact(particle, top):
    phi = abs(particle['geometric parameters']['phi'])
    radius = particle['geometric parameters']['radius']
    

    
    if top and particle['center'][2] + cos(phi)*radius >= simulation.len_z - Z_TOLERANCE:

        return True
    elif not top and particle['center'][2] - cos(phi)*radius <= Z_TOLERANCE:

        return True
    


def calc_volume(particle):
    '''
    calculates the volume of a particle
    '''
    if particle['shape'] == "sphere": 
        radius = particle['geometric parameters']['radius']
        return (4/3)*pi*radius*radius*radius
    
    if particle['shape'] == 'fiber':
        radius = particle['geometric parameters']['radius']
        length = particle['geometric parameters']['length']
        return (4/3)*pi*radius*radius*radius + length*pi*radius**2
    
    if particle['shape'] == 'disk':
        return particle['geometric parameters']['volume']

def calc_vector_length(vector):
    '''
    calculates de length of a vector
    '''
    return sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def calc_distance_3d(pos1,pos2):
    '''
    calculates the distance between two sets of 3 coordinates (x,y,z)
    '''
    
    delta_x = pos1[0] - pos2[0]
    delta_y = pos1[1] - pos2[1]
    delta_z = pos1[2] - pos2[2]
    return sqrt(delta_x**2 + delta_y**2 + delta_z**2)

def calc_distance_2d(pos1,pos2):
    '''
    calculates the distance between two sets of 2 coordinates (x,y)
    '''
    
    delta_x = pos1[0] - pos2[0]
    delta_y = pos1[1] - pos2[1]
    return sqrt(delta_x**2 + delta_y**2)


def calc_squared_distance_3d(pos1,pos2):
    '''
    calculates the square of the distance between two sets of 3 coordinates (x,y,z)
    much faster than calc_distance_3d()
    '''
    delta_x = pos1[0] - pos2[0]
    delta_y = pos1[1] - pos2[1]
    delta_z = pos1[2] - pos2[2]
    return delta_x**2 + delta_y**2 + delta_z**2

def calc_squared_distance_2d(pos1,pos2):
    '''
    calculates the square of the distance between two sets of 3 coordinates (x,y,z)
    much faster than calc_distance_3d()
    '''
    delta_x = pos1[0] - pos2[0]
    delta_y = pos1[1] - pos2[1]
    return delta_x**2 + delta_y**2

def create_particle_template():
    '''
    returns a particle template object form user inputs
    '''
    global particle_templates
    while 1:
        par_list = input_particle_parameters()
        particle_templates.append(ParticleTemplate(par_list['shape'], 
                                                   par_list['distribution'],
                                                   par_list['dist_parameters'],
                                                   par_list['frequency'],
                                                   par_list['conductive']))

        options = OTHER_MESSAGES[lang_index][3]+"/"+OTHER_MESSAGES[lang_index][4]
        while 1:
            answer = input(OTHER_MESSAGES[lang_index][2]+options+"): ")
            if answer == OTHER_MESSAGES[lang_index][4]:
                repeat = True
                break
            elif answer == OTHER_MESSAGES[lang_index][3]:
                repeat = False
                break
        if repeat: break
    
    

def input_particle_parameters():
    '''
    inputs the user for the particle parameters and returns the values as a tuple
    '''
    
    while 1:
        
        #inputs the shape of the particle
        input_str = INPUT_PROMPT[lang_index][0] + str(PARTICLE_SHAPES[lang_index]) + "): "
        particle_shape = input(input_str)
        if particle_shape in PARTICLE_SHAPES[lang_index]:
            #checks if the input is valid, converts to English and breaks
            index =  PARTICLE_SHAPES[lang_index].index(particle_shape)
            particle_shape = PARTICLE_SHAPES[0][index]
            break
           
    while 1:
        
        #inputs the type of particle geometry distribution
        input_str = INPUT_PROMPT[lang_index][1] + str(SIZE_DISTRIBUTIONS[lang_index]) + "): "
        particle_distribution = input(input_str)
        if particle_distribution in SIZE_DISTRIBUTIONS[lang_index]:
            #checks if the input is valid, converts to English and breaks
            index = SIZE_DISTRIBUTIONS[lang_index].index(particle_distribution)
            particle_distribution = SIZE_DISTRIBUTIONS[0][index]
            break
        
    #inputs the type of geometry distribution parameters
    geometries = GEOMETRY_PROMPT[particle_shape][lang_index]
    parameters = DISTRIBUTION_PROMPT[lang_index][particle_distribution]
    #for parameter in [i+" "+j for i in geometries for j in parameters]: #creates a list of parameter and dimension combination
    temp = []
    for geometry in geometries:
        
        for parameter in parameters:
            while 1:
                try:
                    value = input(geometry + " " + parameter + ": ")
                    value = float(value) # throws an error if the input is not a number                
                    break
                except ValueError:
                    print(OTHER_MESSAGES[lang_index][0])
            temp.append(value)
    distribution_parameters = temp
    while 1:
        try:
            value = input(OTHER_MESSAGES[lang_index][1])
            if value == "" : value = 1
            frequency = float(value) # throws an error if the input is not a number
            break
        except ValueError:
            print(OTHER_MESSAGES[lang_index][0])
    
    # defines if particle template is conductive        
    while 1:
        value = input(OTHER_MESSAGES[lang_index][5]+OTHER_MESSAGES[lang_index][6]+" , "+OTHER_MESSAGES[lang_index][7] + ")")
        if value == OTHER_MESSAGES[lang_index][6]:
            conductive = 1
            break
        elif value == OTHER_MESSAGES[lang_index][7]:
            conductive = 0
            break
        print("invalid value")
        

    if DEBUG: print(particle_shape, particle_distribution, distribution_parameters, frequency)

    return {'shape': particle_shape,
            'distribution' : particle_distribution,
            'dist_parameters' : distribution_parameters,
            'frequency' : frequency,
            'conductive' : conductive
            }

def dist_normal(param):
    out = []
    i = 0
    while i < len(param) :
        avg = param[0+i]
        std = param[1+i]
        result = 0
        while result <= 0:
            result = random.normal(avg,std)
        out.append(result)
        i += 2        
        
    return out

def dist_lognormal(param):
    out = []
    i = 0
    while i < len(param):
        loc = param[0+i]
        sca = param[1+i]
        result = param[2+i]
        while result >= param[2+i]:        
            result = random.lognormal(loc,sca)
        out.append(result)
        i += 3
            
    return out

def dist_constant(param):
    return param

def alfa_func(u):
    if u < -1: return -1
    elif u > 1 : return 1
    else: return u

class ConductivityGenerator():
    
    def __init__(self, avg, std, sigma_cla, tunnel_a, tunnel_b, a):
        self.avg = avg
        self.std = std
        self.sigma_cla = sigma_cla
        self.tunnel_a = tunnel_a
        self.tunnel_b = tunnel_b
        self.a = a
        
    def new(self):
        s = self.roll_distance_normal()
        s_tun = 10**(self.tunnel_a-self.tunnel_b*s)*self.a
        s_cla = (self.sigma_cla*self.a)/(s*10**-8)
        if s_tun >= s_cla: return s_tun
        else: return s_cla
        
    def roll_distance_normal(self):
        dist = rd.gauss(self.avg,self.std)
        if dist <= 0: dist = 0.001
        return dist
    
class NetworkSolver():
    
    def __init__(self, raw_network):
        
        self.res_network = self.format_network(raw_network)
        self.simplified = False
        self.remove_nodes()

        
    def format_network(self, raw_network):
        '''
        makes network formatting compatible with algortihm: no resistances connecting the node to itslef,
        summ all resistors that conect qthe same two nodes, every node only registers links to nodes of
        higher index, and all links in a node are ascendingly sorted by their target node index
        '''
        formated_network = []
        for node_index, node in indexed(raw_network):
            new_node = []
            
            for link in node:
                
                if link[0] == 0: break #removes insulated links as they lead to division by zero
                
                if link[1] == node_index:
                    # error: resistor connects node to itself --> remove resistor
                    pass
                elif link[1] < node_index:
                    # error: resistor order is reversed (in larger index, connecting 
                    # to lower index) --> move to correct node
                    formated_network[link[1]].append([link[0],node_index])
                    
                else: #link[1] > node_index: correct link --> add to node
                    new_node.append(list(link))
                    
        
                
            formated_network.append(new_node)
                 

        #add duplicate resistors together
        for node in formated_network:
            targets_set = set([link[1] for link in node])
            if len(targets_set) == len(node): continue
            
            for target in targets_set:
                parallel_links = [link for link in node if link[1] == target]
                
                if len(parallel_links) > 1:
                    new_link = [sum([i[0] for i in parallel_links]),target]
                    for link in tuple(node):
                        if link[1] == target: node.remove(link)
                    node.append(new_link)
                   
          
        # sort links within a node
        for node in formated_network:
            node.sort(key = lambda i:i[1])

           
        return formated_network

        
    def increment_link(self, node_a, node_b, delta_g):
        

        if node_a == node_b:
            print('error')
            0/0
        
        # makes sure node_a < node_b    
        if node_a > node_b: node_a, node_b = node_b, node_a
        
        repeated_link = [i for i in self.res_network[node_a] if i[1] == node_b]

        
        if len(repeated_link) > 1:
            print('error')
            0/0
        
        if len(repeated_link) == 0:
            self.res_network[node_a].append([delta_g,node_b])
            
        else:
            delta_g += repeated_link[0][0]
            self.res_network[node_a].remove(repeated_link[0])
            self.res_network[node_a].append([delta_g,node_b])

        
    def remove_nodes(self):
        


        for node_index, node in indexed(self.res_network[0:-2]):
            
            if len(node) <= 1: 
                self.res_network[node_index] = []
                pass # discard "deadend" and isolated nodes
            
            g_sum = sum([i[0] for i in node])
            linked_pairs = itertools.combinations([i for i in node if i[1] > node_index],2)
            #print "node", node_index, "of", len(self.res_network), "/", len([i for i in node if i[1] > node_index])**2, "resistors"
            
            for pair in linked_pairs:
                delta_g = (pair[0][0]*pair[1][0])/g_sum
                self.increment_link(pair[0][1], pair[1][1], delta_g)
                
            self.res_network[node_index] = []

        
        self.simplified = True 
        
    def get_conductivity(self):
            try:
                return self.res_network[-2][0][0]
            except IndexError:
                return 0

# defines the simulation object
class Simulation():
    
    # main method, performs the simulation
    def simular(self,target_filler_fraction,len_x,len_y,len_z):
        '''
        variables initialization
        '''
        self.cluster = []
        self.len_x = len_x
        self.len_y = len_y
        self.len_z = len_z
        self.target_filler_fraction = target_filler_fraction
        self.contacts = None
        self.res_network=None
        self.percolation = 0
        
        
        # creates a list of templates frequencies
        self.template_frequency_threshold = [particle_templates[0].get_frequency()]
        for template in particle_templates[1:]:
            self.template_frequency_threshold.append(template.get_frequency())
        
        # make frequencies cumulative   
        self.template_frequency_threshold = [sum(self.template_frequency_threshold[:i+1]) for i in range(len(self.template_frequency_threshold))]
        
        # normalizes cumulative frequencies
        self.template_frequency_threshold = [i / self.template_frequency_threshold[-1] for i in self.template_frequency_threshold]
        
        
        if DEBUG:    
            for i in particle_templates:
                print(i)
        
        # keeps track of successful percolations in multiple simulations
        global percolations_total
        
        # creates the sectors for this simulation
        self.sectors = SectorsAlfa(len_x,len_y,len_z)
        
        # creates a size array for the quick-union algorithm

        
        # the particle library keeps tracks of each individual sphere
        '''
        the properties indexes are:
        0: geometry of the particle
        1: center coordinates of the particle (x,y,z)
        2: dictionary of geometric parameters
        '''
        
        self.particle_library = []
        

        #fills the particle library
        self.create_all_particles()
        
        if VERBOSE_DEBUG: print(self.particle_library)
        
        #acomodates the particles according to the hardcore model
        global file_number
        global moved_particles
        if HARDCORE:
            moved_particles = 0
            # self.link_roots()
            while self.acomodate_particles():
                moved_particles = 0
            
        print(file_number, )
        file_number = 0
        
        self.link_roots()    

        return self.detect_percolation()
    
    def record_contacts(self):
        
        if self.contacts != None:
            return
        
        self.contacts = []
        
        '''
        create a list of all the contacts between particles
        '''

        self.sectors.empty_sectors()
        
        for active_particle in self.particle_library:
        
            new_node = []

            if active_particle['conductive'] == 0 :
                self.contacts.append(new_node)
                continue
            active_root = self.root(active_particle)
           
            
            for target_particle in self.particle_library[active_particle['index']+1:]:
                if target_particle['conductive']:
                    if active_root == self.root(target_particle):
                        if check_contact(active_particle, target_particle, override_conductivity = 1):
                            contact_vector = calc_contact_vector(active_particle,target_particle)
                            new_node.append([contact_vector,target_particle['index']])
                    
            self.contacts.append(new_node)
        
        
        #creates link to the z planes
        new_node = []
        for particle in self.particle_library:
            if check_contact_plane(particle,'z',False):
                new_node.append([(1,1,1),particle['index']])
        self.contacts.append(new_node)
        
        new_node = []
        for particle in self.particle_library:
            if check_contact_plane(particle,'z',True):
                new_node.append([(1,1,1),particle['index']])
        self.contacts.append(new_node)
                    
    def create_distance_network(self, dist_avg, dist_std, flat_dist):
        '''
        takes the contacts list of the simulated composite and makes a list of resistance
        from this list. the tension and poisson coefficients are given to simulate strain
        and the electrical properties of the system, specifically the conductivity of the
        matrix, the contact area and the parameters of the tunneling approximation
        '''

        distance_network = []
        
        for node in self.contacts:
            new_node = []
            for link in node:
                if not flat_dist:
                    distance = roll_distance_normal(dist_avg,dist_std)
                elif flat_dist:
                    distance = roll_distance_flat(1,10)
                new_node.append([link[0],distance,link[1]])
            distance_network.append(new_node)

        return distance_network
                
    def deform_network(self, tension, poisson, input_network):
        '''
        takes an input network and distorts it before returning it
        '''
        #convert tension to strain, assuming E = 1:
        strain = (tension[0] - poisson*tension[1] - poisson*tension[2],
                  -poisson*tension[0] + tension[1] - poisson*tension[2],
                  -poisson*tension[0] -poisson*tension[1] + tension[2])
        
        strained_network = []
        
        for node in input_network:
            new_node = []
            for link in node:
                distance = link[1]
                contact_vector = link[0]
                distorted_vector = [i*(1-j) for i,j in zip(contact_vector, strain)]
                distorted_distance = distance * calc_vector_length(distorted_vector)
                new_node.append([distorted_distance, link[2]])
            strained_network.append(new_node)
        
        return strained_network
    
    def create_resistence_network(self, tension, poisson, classic_sigma, contact_area, 
                                  tunnel_a, tunnel_b, distance_network, flat_dist):
        
        input_network = self.deform_network(tension, poisson, distance_network)
        
        res_network = []
        
        for node in input_network:
            new_node = []
            for link in node:
                conductivity = S(link[0], classic_sigma,tunnel_a, tunnel_b, contact_area)
                new_node.append([conductivity, link[1]])
            res_network.append(new_node)
        
        highest = max([link[0] for node in res_network for link in node])
        
        for link in res_network[-2]:
            link[0] = highest * 100 
            
        for link in res_network[-1]:
            link[0] = highest * 100    
            
        return res_network
        
    def simulate_conductivity(self, tension_list, poisson, classic_sigma, contact_area, tunnel_a, tunnel_b,
                              dist_avg, dist_std, flat_dist = False):
        
        
        conductivity_results = []        
        if self.percolation == 0:
            conductivity = S(10, classic_sigma, tunnel_a, tunnel_b, 1)
            conductivity_results.append([conductivity,[0,0,0]])
            for tension in tension_list[1:]:
                conductivity_results.append([conductivity,tension])
            return conductivity_results
        
        self.record_contacts()
        distance_network = self.create_distance_network(dist_avg, dist_std, flat_dist)
        
        
        
        for tension in tension_list:
            
            res_network = self.create_resistence_network(tension, poisson, classic_sigma, 
                                                         contact_area, tunnel_a, tunnel_b,
                                                         distance_network, flat_dist)
            
            ns = NetworkSolver(res_network)
            conductivity = ns.get_conductivity()
            
            if conductivity == 0: 
                # if the solver can't solve the network, returns and insulating network
                # Rare occurrence when the hardcore algorithm thrusts a particle out of
                # the contact range                
                conductivity = S(10, classic_sigma, tunnel_a,tunnel_b,1)
                conductivity_results.append([conductivity,[0,0,0]])
                for tension in tension_list[1:]:
                    conductivity_results.append([conductivity,tension])
                return conductivity_results
            
            conductivity_results.append([conductivity, tension])
            
        return conductivity_results
        
    def unite(self,particle1,particle2):
        

        if particle1['tree size'] >= particle2['tree size']:
            self.particle_library[self.root(particle1)]['root'] = self.root(particle2)
        else:
            self.particle_library[self.root(particle2)]['root'] = self.root(particle1)
            
    
    def find(self,particle1,particle2):
        return self.root(particle1) == self.root(particle2)
    
    def root(self,starting_particle):
        size = 1
        particle = starting_particle
        while particle['index'] != particle['root']:
            particle['root'] = self.particle_library[particle['root']]['root']
            particle = self.particle_library[particle['root']]
            size +=1
            
        starting_particle['tree size'] = size
        return particle['index']
    
    def create_all_particles(self):
        
        filler_volume = 0
        target_filler_volume = self.target_filler_fraction * self.len_x * self.len_y * self.len_z
        index = 0
        
        # Adds a counter of added particles of each shape and conductivity, that can be printed for debugging purposes
        template_count = {}
        for template in particle_templates:
            template_count[template.shape + str(template.conductive)] = 0
            
        while filler_volume < target_filler_volume:
            
            chosen_template = -1
            # choose the template
            random_number = random.random()
            
            for frequency in self.template_frequency_threshold:
                if random_number <= frequency and chosen_template == -1: chosen_template = particle_templates[self.template_frequency_threshold.index(frequency)]
                
            template_count[chosen_template.shape + str(chosen_template.conductive)] += 1 #updates the informative counter
                
            # append the new particle to the library
            index = len(self.particle_library)        
            self.particle_library.append(chosen_template.add_particle(self.len_x, self.len_y, self.len_z, index))
                        
            # update filler_volume
            
            filler_volume += calc_volume(self.particle_library[index])           
            index += 1

        print('number of particles: ' + str(index))
        
        # print template_count
        
    def acomodate_particles(self):
        '''
        Code that accommodate particles
        returns false if accommodation is complete, or true if another iteration must be run
        '''
        global file_number, moved_particles
        
        current_soft_shell =  1 - (1-SOFT_SHELL)**((file_number // HARDCORE_TOLERANCE)+1)
                
        if file_number < 11 and OUTPUT:
            blender_file(file_number)
            print('file saved', file_number)
            
        if file_number % 200 == 0 and file_number!= 0 and OUTPUT:
            blender_file(file_number)
            print('file saved', file_number)
            
        file_number += 1

            
        if file_number > HARDCORE_INTERRUPT :
            print('interrupted accommodation')
            return False
        
        '''
        implements the hard core model
        '''
        self.sectors.empty_sectors() 
            
        for particle in self.particle_library:
            particle['displacement vectors'] = []
            

        
        #fills the displacement vectors list
        for active_particle in self.particle_library:
            
            
            if active_particle['shape'] == 'sphere' or active_particle['shape'] == 'fiber':
                radius = active_particle['geometric parameters']['radius']
                
                # check_borders:
                
                #x axis
                if active_particle['center'][0] - radius*(1-current_soft_shell) < 0 :
                    active_particle['displacement vectors'].append([(radius - active_particle['center'][0])*(1-current_soft_shell),0,0])
                elif active_particle['center'][0] + radius*(1-current_soft_shell) > self.len_x :
                    active_particle['displacement vectors'].append([(-radius + (self.len_x-active_particle['center'][0]))*(1-current_soft_shell),0,0])
                    
                #y axis
                if active_particle['center'][1] - radius*(1-current_soft_shell) < 0 :
                    active_particle['displacement vectors'].append([0,(radius - active_particle['center'][1])*(1-current_soft_shell),0])
                elif active_particle['center'][1] + radius*(1-current_soft_shell) > self.len_y :
                    active_particle['displacement vectors'].append([0,(-radius + (self.len_y-active_particle['center'][1]))*(1-current_soft_shell),0])
                    
                #z axis
                if active_particle['center'][2] - radius*(1-current_soft_shell) < 0 :
                    active_particle['displacement vectors'].append([0,0, (radius - active_particle['center'][2])*(1-current_soft_shell)])
                elif active_particle['center'][2] + radius*(1-current_soft_shell) > self.len_z :
                    active_particle['displacement vectors'].append([0,0,(-radius + (self.len_y-active_particle['center'][2]))*(1-current_soft_shell)])
                    
                # check other particles in contact
                indexes_in_sector = self.sectors.add_particle_to_sectors(active_particle)
                for target_particle in [self.particle_library[index] for index in indexes_in_sector]:

                    
                    if check_contact(active_particle,target_particle,current_soft_shell, override_conductivity = 1) and active_particle != target_particle:
                        
                        
                        volume1 = calc_volume(active_particle)
                        volume2 = calc_volume(target_particle) 
                                       
                        total_volume = volume1 + volume2
                            
                        volume1 = volume1 / total_volume
                        volume2 = volume2 / total_volume
               
                        
                        if active_particle['shape'] == 'sphere' and target_particle['shape'] == 'sphere':
                          
                            direction_vect = [j-i for (i,j) in zip(active_particle['center'],target_particle['center'])]
                            distance = calc_distance_3d(active_particle['center'],target_particle['center'])
                            interpenetrating_distance = radius + active_particle['geometric parameters']['radius'] - distance
                            
                            for i in range(3):
                                direction_vect[i] = direction_vect[i] * interpenetrating_distance

                            active_particle['displacement vectors'].append([direction_vect[0]*-volume2,direction_vect[1]*-volume2,direction_vect[2]*-volume2])
                            target_particle['displacement vectors'].append([direction_vect[0]*volume1,direction_vect[1]*volume1,direction_vect[2]*volume1])
                        
                        elif active_particle['shape'] == 'fiber' and target_particle['shape'] == 'fiber':
                            direction_vect = fiber_fiber_contact(active_particle, target_particle, current_soft_shell, return_vector = True)
                            active_particle['displacement vectors'].append([direction_vect[0]*-volume2,direction_vect[1]*-volume2,direction_vect[2]*-volume2])
                            target_particle['displacement vectors'].append([direction_vect[0]*volume1,direction_vect[1]*volume1,direction_vect[2]*volume1])
                            
                        elif active_particle['shape'] == 'sphere' and target_particle['shape'] == 'fiber':
                            direction_vect = sphere_fiber_contact(active_particle, target_particle, current_soft_shell, return_vector = True)
                            active_particle['displacement vectors'].append([direction_vect[0]*-volume2,direction_vect[1]*-volume2,direction_vect[2]*-volume2])
                            target_particle['displacement vectors'].append([direction_vect[0]*volume1,direction_vect[1]*volume1,direction_vect[2]*volume1])
                            
                        elif active_particle['shape'] == 'fiber' and target_particle['shape'] == 'sphere':
                            direction_vect = sphere_fiber_contact(target_particle, active_particle, current_soft_shell, return_vector = True)
                            active_particle['displacement vectors'].append([direction_vect[0]*volume2,direction_vect[1]*volume2,direction_vect[2]*volume2])
                            target_particle['displacement vectors'].append([direction_vect[0]*-volume1,direction_vect[1]*-volume1,direction_vect[2]*-volume1])
                            
                        else: 1/0
                            
        # end of particle displacement vectors list
        
        
        # move the particles
        modified = False       
        
        for active_particle in self.particle_library:
            
            vectors = active_particle['displacement vectors']
            
            if len(vectors) == 1:
                
                for i in range(3):
                    active_particle['center'][i]+=(vectors[0][i] * ACCOMMODATION_STEP)
                moved_particles += 1
                modified = True
                
            elif len(vectors) > 1:
                vector_lens = []
                for vector in vectors: vector_lens.append(calc_distance_3d((0,0,0),vector))
                
                # add the vectors
                added_vector = [0,0,0]                
                for vector in vectors:
                    added_vector[0] += vector[0]
                    added_vector[1] += vector[1]
                    added_vector[2] += vector[2]
                    

                for i in range(3):
                    active_particle['center'][i] += ((added_vector[i]/len(vectors)) * ACCOMMODATION_STEP)
                
                moved_particles += 1    
                modified = True
                
                
        if modified:
            return True
        else:
            return False
                                    
    def clean_roots(self):
        
        for particle in self.particle_library: 
            particle['root'] = particle['index']
    
    def link_roots(self):
        '''
        create the links between particles checking contact between particles in the same sector
        '''

        self.sectors.empty_sectors()
        
        for active_particle in self.particle_library:
        
            particles_in_sector = self.sectors.add_particle_to_sectors(active_particle)
                        
            # make contact checks between the just-added particle
            # and the particles found in the same sectors
            
            for target_particle in [self.particle_library[index] for index in particles_in_sector]:
                if check_contact(active_particle,target_particle):
                    self.unite(active_particle,target_particle) 
                              
        for particle in self.particle_library:  
            self.root(particle)
                 
            
            
    def detect_percolation(self):
        '''
        percolation is currently always detected across the z axis
        '''
        
        number_of_sectors = self.sectors.get_sectors_in_axis()
        sectors_x = number_of_sectors[0]
        sectors_y = number_of_sectors[1]
        sectors_z = number_of_sectors[2]
        indexes_in_z0 = []
        indexes_in_zmax = []
        roots_in_z0 = []
        roots_in_zmax = []
        

        i = 0
        j = 0        
        while i < sectors_x:
            while j < sectors_y:
                indexes_in_z0.extend(self.sectors.get_indexes_in_sector([i,j,0]))
                j += 1
            j = 0
            i += 1
                    
        i = 0
        j = 0
        while i < sectors_x:
            while j < sectors_y:

                indexes_in_zmax.extend(self.sectors.get_indexes_in_sector([i,j,sectors_z-1]))
                j += 1
            j = 0
            i += 1
            
        indexes_in_z0 = list(set(indexes_in_z0))
        indexes_in_zmax = list(set(indexes_in_zmax))
        
        for particle in [self.particle_library[index] for index in indexes_in_z0]:
            if check_contact_plane(particle,'z',False):
                roots_in_z0.append(self.root(particle))
            
        for particle in [self.particle_library[index] for index in indexes_in_zmax]:
            if check_contact_plane(particle,'z',True):
                roots_in_zmax.append(self.root(particle))
            
        roots_in_z0 = (list(set(roots_in_z0)))
        roots_in_zmax = (list(set(roots_in_zmax)))
        
        
        for i in roots_in_z0:
            if i in roots_in_zmax: 
                self.percolation = 1
                self.cluster.append(i)
                   
        self.calc_interconectivity()
        
        
        return (self.percolation, self.interconectivity)
                                                                       
            
    def calc_interconectivity(self):
        self.interconectivity = 0
        
        for particle in self.particle_library:
            if self.root(particle) in self.cluster:
                self.interconectivity += 1
        
                
        self.interconectivity = self.interconectivity / len(self.particle_library)
        
                

# the Sectors class deals with the sector optimization
# called alfa to allow several version for further performance evaluation
class SectorsAlfa():
    
    def __init__(self,len_x,len_y,len_z):
        
        # calculates the highest maximum radius (within 95% confidence)
        radius = []
        for template in particle_templates:
            if template.get_shape() == 'sphere':
                if template.get_distribution() == 'constant' : radius.append(template.get_parameters()[0])
                elif template.get_distribution() == 'normal' : radius.append(template.get_parameters()[0]+2*template.get_parameters()[1])
                elif template.get_distribution() == 'lognormal' : radius.append(math.exp(template.get_parameters()[0] + template.get_parameters()[0]**2/2))
                
            elif template.get_shape() == 'fiber':
                if template.get_distribution() == 'constant' : radius.append(template.get_parameters()[1]/2+template.get_parameters()[0])
                elif template.get_distribution() == 'normal' : radius.append((template.get_parameters()[2]+2*template.get_parameters()[3])/2+(template.get_parameters()[0]+2*template.get_parameters()[1]))
                elif template.get_distribution() == 'lognormal' : radius.append(math.exp(template.get_parameters()[2] + template.get_parameters()[3]**2/4)+math.exp(template.get_parameters()[0] + template.get_parameters()[1]**2/2))
                
            elif template.get_shape() == 'disk':
                if template.get_distribution() == 'constant' : radius.append(template.get_parameters()[0])
                elif template.get_distribution() == 'normal' : radius.append(template.get_parameters()[0]+2*template.get_parameters()[1])
                elif template.get_distribution() == 'lognormal' : radius.append(math.exp(template.get_parameters()[0] + template.get_parameters()[0]**2/2))
                
            else:
                print("unknown particle geometry")
            
        self.sector_size = 2*max(radius)
        
        #calculates number of sector in each axis:
        self.sectors_in_x = int(math.ceil(len_x/self.sector_size))
        self.sectors_in_y = int(math.ceil(len_y/self.sector_size))
        self.sectors_in_z = int(math.ceil(len_z/self.sector_size))

        # creates the empty sectors
        i=0
        j=0
        k=0
        self.indexes_in_sector = []
        while i < self.sectors_in_x:
            self.indexes_in_sector.append([])
            while j < self.sectors_in_y:
                self.indexes_in_sector[i].append([])
                while k < self.sectors_in_z:
                    self.indexes_in_sector[i][j].append([])
                    k += 1
                k = 0
                j += 1
            i += 1
            j = 0
            
    def add_particle_to_sectors(self, particle):
        '''
        adds the index of the sphere at the coordinates and of the radius 
        to the the appropriate sectors, also returns a list of non-repeating
        indexes of the other particles already added to the sectors where
        the added sphere is present        
        '''
        center = particle['center']
        index = particle['index']
        
        if particle['shape'] == 'sphere':
            contact_range = particle['geometric parameters']['radius']
        elif particle['shape'] == 'fiber':
            contact_range = particle['geometric parameters']['length']/2 + particle['geometric parameters']['radius']
        elif particle['shape'] == 'disk':
            contact_range = particle['geometric parameters']['radius']
        
        #defines high and low sectors in each axis
        high_x = min(int(math.ceil((center[0]+contact_range)/self.sector_size)),self.sectors_in_x-1)
        low_x = max(int(math.floor((center[0]-contact_range)/self.sector_size)),0)
        high_y = min(int(math.ceil((center[1]+contact_range)/self.sector_size)),self.sectors_in_y-1)
        low_y = max(int(math.floor((center[1]-contact_range)/self.sector_size)),0)
        high_z = min(int(math.ceil((center[2]+contact_range)/self.sector_size)),self.sectors_in_z-1)
        low_z = max(int(math.floor((center[2]-contact_range)/self.sector_size)),0)
        
        candidate_indexes = []
   
        for (x,y,z) in list(itertools.product(range(low_x,high_x+1),range(low_y,high_y+1),range(low_z,high_z+1))):
            try:
                candidate_indexes.extend(self.indexes_in_sector[x][y][z])
                self.indexes_in_sector[x][y][z].append(index)                        
            except IndexError:
                print(x,y,z)
                if len(self.sectors[x][y][z]) == 1 :
                    candidate_indexes.append(self.indexes_in_sector[x][y][z])
                self.indexes_in_sector[x][y][z].append(index)
            
   
        return list(set(candidate_indexes))
    
    def empty_sectors(self):
        
        for (i,j,k) in [(x,y,z) 
                        for x in range(self.sectors_in_x) 
                        for y in range(self.sectors_in_y) 
                        for z in range(self.sectors_in_z)]:
        
            self.indexes_in_sector[i][j][k]=[]

            
    def get_indexes_in_sector(self,coordinates):
        
        return self.indexes_in_sector[coordinates[0]][coordinates[1]][coordinates[2]]
    
    def get_sectors_in_axis(self):
        
        return([self.sectors_in_x,self.sectors_in_y,self.sectors_in_z])
        
    
# the particle template keeps track of the particles characteristics and procedures
class ParticleTemplate():
    
    def __init__(self, shape, size_distribution, parameters, frequency, conductive):
        self.shape = shape
        self.distribution = size_distribution
        self.parameters = parameters
        self.frequency = frequency
        self.conductive = conductive
        print(parameters)
        
        
        #checks for the geometry values distribution
        if self.distribution == "normal":
            self.get = dist_normal
        elif self.distribution == "lognormal":
            self.get = dist_lognormal
        elif self.distribution == "constant":
            self.get = dist_constant

        #checks geometric parameters

        self.geometry = []
        if self.shape == "sphere":
            self.geometry.append("radius")
        if self.shape == "fiber":
            self.geometry.append("radius")
            self.geometry.append("length")
        if self.shape == "disk":
            self.geometry.append("radius")
            self.geometry.append("length")
            
        #assign values to parameters
        self.dict_geo = {}
        for geo in self.geometry:
            self.dict_geo[geo] = self.parameters
            
            
    def __str__(self):
        return self.shape + str(self.distribution) + str(self.parameters) + str(self.frequency) + str(self.dict_geo)
        
    def add_particle(self, len_x, len_y, len_z, index):
        '''
        generates a dictionary with particles characteristics
        output:
        {'shape'} -> with particle shape (string)
        {'center'} -> coordinates of mass center (tupple with three floats representing x,y,z)
        {'geometric_parameters'} -> dictionary with geometric parameter depending on particle shape
        {'conductive'} -> is the particle conductive?
        '''
        output = {}
        
        output['index'] = index
        
        output['root'] = index
        
        output['shape'] = self.shape
        
        output['tree size'] = 1
        
        output['center'] = [random.random(),random.random(),random.random()]

        output['geometric parameters'] = {}
        
        output['color'] = 0
        
        generated_geometries = self.get(self.parameters) 
        i = 0
        for geo in self.geometry:
            output['geometric parameters'][geo] = generated_geometries[i] #2: specific parameters 
            i += 1
            

        if self.shape == 'sphere':
            radius = output['geometric parameters']['radius']
            output['center'][0] = output['center'][0] * (len_x-2*radius) + radius
            output['center'][1] = output['center'][1] * (len_y-2*radius) + radius
            output['center'][2] = output['center'][2] * (len_z-2*radius) + radius
            
        if self.shape == 'fiber':
            
            radius = output['geometric parameters']['radius']
            half_length = output['geometric parameters']['length']/2
            
            output['center'][0] = output['center'][0] * (len_x-2*radius) + radius
            output['center'][1] = output['center'][1] * (len_y-2*radius) + radius
            output['center'][2] = output['center'][2] * (len_z-2*radius) + radius
            
            theta = random.uniform(-MAX_ANGLE_X, MAX_ANGLE_X)*pi/2 # angle limitation is between 0 and 1 
            phi = arccos(random.uniform(-1,1)) # angle limitations are between 0 and 1
            
            x = half_length * sin(phi) * cos(theta)
            y = half_length * sin(phi) * sin(theta)
            z = half_length * cos(phi)
            
            output['geometric parameters']['half shaft'] = (x,y,z)
            output['geometric parameters']['rotation'] = (phi,0,theta+pi/2)
            
        if self.shape == 'cylinder':
            if CYLINDER_AS_DISK:
                output['shape'] = 'disk'
                radius = output['geometric parameters']['radius']
                length = output['geometric parameters']['length']
                output['center'][0] = output['center'][0] * (len_x-2*length) + length
                output['center'][1] = output['center'][1] * (len_y-2*length) + length
                output['center'][2] = output['center'][2] * (len_z-2*length) + length
                equivalent_radius = radius * (1 + 2.85*(length/radius))**(1/3)
                output['geometric parameters'] = {'radius' : equivalent_radius }
                theta = random.uniform(-pi,pi)/2
                phi = arccos(random.uniform(-1,1))
                x = sin(phi) * cos(theta)
                y = sin(phi) * sin(theta)
                z = cos(phi)
                output['geometric parameters']['normal vector'] = (x,y,z) # always a unit vector
                output['geometric parameters']['phi'] = phi
                output['geometric parameters']['volume'] = length*pi*radius**2
            else: 
                print(self.shape + ' shape not found')
                0/0
                
        output['conductive'] = self.conductive       
               
        return output
    
    def get_frequency(self):
        return self.frequency
    
    def set_frequency(self,new_value):
        self.frequency = new_value
        
    def get_shape(self):
        return self.shape
    
    def get_distribution(self):
        return self.distribution
    
    def get_parameters(self):
        return self.parameters
    
# lines to run before start of simulation

#ns = NetworkSolver([[], [], [[2.2799382843767639e-08, 1]], [[1.1049149400330103e-05, 0]], [], [], [[1.2275260138564855e-06, 5]], [[0.0011705414622118688, 1], [0.0016769982642624716, 2]], [], [], [], [[8.6508764354193847e-10, 5], [6.1920282771728796e-06, 6]], [], [[4.9751020263745274e-08, 1], [4.4789364035352677e-09, 2], [5.5943536803350586e-05, 7]], [], [], [], [], [], [], [[6.7800688564664279e-05, 1], [4.3753434027500781e-06, 2], [0.0087249553621176687, 7], [3.1771312042637102e-07, 13]], [], [], [[1.22016049170912e-07, 17]], [[2.8901057755535312e-05, 0]], [[0.014026577739092848, 1], [2.5755338124621274e-08, 2], [0.0001134022532465809, 7], [0.010400742915284339, 13], [0.0010020043747647883, 20]], [[1.5208919155156466, 11]], [[1.5208919155156466, 0], [1.5208919155156466, 16]]])     
#print ns.res_network
#ns.get_conductivity()
            

# initiates the program:

simulation = Simulation()

if DEBUG:
    particle_templates.append(ParticleTemplate('sphere', 'constant', 1, 4))
    #particle_templates.append(ParticleTemplate('sphere', 'normal', [0.5,1], 1))
    #particle_templates.append(ParticleTemplate('sphere', 'normal', [1.63, 1.46], 1))
else:    
    create_particle_template()


'''
temporary input lines
'''

HARDCORE = 0
concentrations = [ 0.35 - i * 0.0025 for i in range(1) ]
axis_proportion = [100, 100, 100]
base_lens = [ 1]
x_lens = [axis_proportion[0]*i for i in base_lens]
y_lens = [axis_proportion[1] for i in base_lens]
z_lens = [axis_proportion[2]*i for i in base_lens]
individual_iteration = 1
tensions_list = ()#([0,0,0],[0,0,0.1],[0,0,0.5],[0,0.1,0],[0,0.5,0])
run_name = 'Fc '
if HARDCORE == 0: run_name += ' SC'
TOTAL_ITERATIONS = len(concentrations) * len(base_lens) * individual_iteration

'''
end of temporary input lines
'''

if len(tensions_list) == 0:
    CALCULATE_CONDUCTIVITY = 0
else:
    CALCULATE_CONDUCTIVITY = 1

smaller_dimensions = []
for template in particle_templates:
    smaller_dimensions.append(min(template.get_parameters()))
    
Z_TOLERANCE = max(smaller_dimensions)

        

iteration_parameters = [(i,x_lens[j],y_lens[j],z_lens[j]) for i in concentrations 
                        for j in range(len(base_lens))]
print(iteration_parameters)

cur_time = time.gmtime()
formated_time = str(cur_time[2]) + '_' + str(cur_time[1]) + '_' + str(cur_time[0])[2:4] + ' ' + str(cur_time[3])  + str(cur_time[4]) + str(cur_time[5]) + ' ' + run_name
print(formated_time)

directory = os.getcwd()
out_file = open(directory + '\\' + formated_time + '.txt', 'w')
out_file.write('Number of individual iterations: ' + str(individual_iteration) + '\n')
out_file.write(' X axis: ' + str(x_lens[0]) + '; Y axis: ' + str(y_lens[0]) + '; Z axis: ' + str(z_lens[0]) + '\n')
out_file.write('Alignment: ' + str((MAX_ANGLE_X, MAX_ANGLE_Z, MIN_ANGLE_Z)) + '\n')

for template in particle_templates:
    out_file.write(str(template) + '\n\n')

out_file.write('Concentration \t Percolation \t Percolation probability \t Interconnectivity \n')



total_conductivities = []

completed_simulations = 0

for param in iteration_parameters:
    i = 0
    total_percolations = 0
    total_interconectivity = 0
    
    conductivities=[]
    while i < individual_iteration:
        
        simulation = Simulation()
        result = simulation.simular(param[0], param[1], param[2], param[3])
    
        total_percolations += result[0]
        total_interconectivity += result[1]
        
        if len(tensions_list) > 0:
            new_cond = simulation.simulate_conductivity(tension_list=tensions_list, poisson = 0.35, classic_sigma=10**-(16.59), contact_area=1, tunnel_a=-0.187, tunnel_b=0.933, dist_avg=10, dist_std=1, flat_dist=True)
            conductivities.append(new_cond)
        
        completed_simulations += 1
        estimated_time = ((time.clock()/completed_simulations)*(TOTAL_ITERATIONS - completed_simulations))/360
        print('ETA: ', estimated_time, 'Hours')
    
        i += 1

    conductivities = [i for i in conductivities if i != 0]
    conductivity_difference = []
    cond_avg = []

    
    if conductivities != []:
        for cond in conductivities:
            new_entry = [cond[0]]
            for strain in cond[1:]:
                new_entry.append([(strain[0]-cond[0][0])/cond[0][0],strain[1]])
            conductivity_difference.append(new_entry)
    
        for i in range(len(conductivity_difference[0])):
            if i == 0 :
                print( sum([log(j[i][0],10) for j in conductivity_difference]) / individual_iteration)
                ca = 10**( sum([log(j[i][0],10) for j in conductivity_difference]) / individual_iteration )
                ca_std = std([log(j[i][0],10) for j in conductivity_difference])
                print(ca)

            else:
                ca = (sum([j[i][0] for j in conductivity_difference])/individual_iteration)
                if ca != (sum([abs(j[i][0]) for j in conductivity_difference])/individual_iteration): print('opposite behaviours')
                ca_std = std([j[i][0] for j in conductivity_difference])                

            cond_avg.append((ca,conductivity_difference[0][i][1],ca_std))
            
        total_conductivities.append((param[0],cond_avg))
    

    print('')  
    if total_percolations > 0:
        total_interconectivity = total_interconectivity / total_percolations
    else:
        total_interconectivity = 0
    
    print('concentration: '+ str(param[0]) + '; iterations: ' + str(individual_iteration) + '; percolations: ' + str(total_percolations) + '; interconectivity: ' + str(total_interconectivity) + '\n\n')
    percolation_ratio_str = '%.4f' % (total_percolations/individual_iteration)
    out_file.write(str(param[0]) + '\t' + str(total_percolations) + '\t' + percolation_ratio_str + '\t' + str(total_interconectivity) + '\n')
        
'''
creates a table of conductivities at the bottom of the file
'''
concentrations = [i[0] for i in total_conductivities]
if concentrations != []:
    strains = [i[1] for i in total_conductivities[0][1]]

    # adds values of average conductivity  
  
    strains_text = '\n\n\t'
    for strain in strains:
        strains_text += str(strain) + '\t'
    strains_text += '\n'
    
    for concentration in total_conductivities:
        strains_text += str(concentration[0])
        for value in concentration[1]:
            strains_text += '\t' + str(value[0])
        strains_text += '\n'
    
    strains_text += '\n'
    
    # adds values of conductivity standard deviation
    strains_text += '\n\n\t'
    for strain in strains:
        strains_text += str(strain) + '\t'
    strains_text += '\n'
    
    for concentration in total_conductivities:
        strains_text += str(concentration[0])
        for value in concentration[1]:
            strains_text += '\t' + str(value[2])
        strains_text += '\n'
    
    strains_text += '\n'
    
    print(strains_text)

if OUTPUT:
    blender_file('final')

elapsed_time = str(time.clock()-ini)
print(elapsed_time)

if CALCULATE_CONDUCTIVITY:
    out_file.write(strains_text)
out_file.write('\n' + elapsed_time)
out_file.close



# script model to create objects in blender through scripts

'''
import bpy
import math

mat = []

MAX_ROOT = 329
CR_ROOT = math.ceil(MAX_ROOT**(1/3))

for i in range(MAX_ROOT):
    mat.append(bpy.data.materials.new('mat'+str(i)))
    color = ((i%CR_ROOT)/CR_ROOT,((i//CR_ROOT)%CR_ROOT)/CR_ROOT,(
                            (i//CR_ROOT**2)%CR_ROOT)/CR_ROOT)
    mat[i].diffuse_color = color


def add_sphere(origin, radius, root):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=7, ring_count=6, size = radius, location=origin)
    ob = bpy.context.object

    ob.active_material = mat[root]
    
def add_cylinder(origin, length, radius, rotation, root):
    
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=8, radius=radius, depth = length, end_fill_type = 'TRIFAN',
        location = origin, rotation = rotation)
    ob = bpy.context.object

    ob.active_material = mat[root]
'''
