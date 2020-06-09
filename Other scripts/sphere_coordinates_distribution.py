# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:03:22 2019

@author: Rafael Arenhart
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

'''
use '%matplotlib qt5' no console IPython para abrir o gráfico interativo
'''

PI = np.pi
SAMPLES = 1000

theta = np.random.random(SAMPLES) * 2 * PI
polar_rho = np.random.random(SAMPLES) * 2 * PI
uniform_rho = np.arccos(2*np.random.random(SAMPLES) - 1)

def polar_to_cartesian(theta, rho, radius=1):

	x = np.sin(rho) * np.cos(theta)
	y = np.sin(rho) * np.sin(theta)
	z = np.cos(rho)

	return (x, y, z)


fig = plt.figure()
axs = [fig.add_subplot(121, projection='3d'),
	   fig.add_subplot(122, projection='3d')]
axs[0].set_axis_off()
axs[0].set_title('Concentrado')
axs[1].set_axis_off()
axs[1].set_title('Distribuído')

polar_points = polar_to_cartesian(theta, polar_rho)
#ax.scatter(polar_points[0], polar_points[1], polar_points[2])
#plt.show()

uniform_points = polar_to_cartesian(theta, uniform_rho)
#ax.scatter(uniform_points[0], uniform_points[1], uniform_points[2])
#plt.show()

#fig, axs = plt.subplots(2)


axs[0].scatter(polar_points[0], polar_points[1], polar_points[2])
axs[1].scatter(uniform_points[0], uniform_points[1], uniform_points[2])
plt.show()
