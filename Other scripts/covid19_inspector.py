# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:56:05 2020

@author: Rafael Arenhart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import requests
import io
import datetime
import csv
import urllib

if True:
	url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
	site = requests.get(url).content
	data = pd.read_csv(io.StringIO(site.decode('utf-8')))
	data = data[data.Confirmed >= 100]
	data = data[data.Deaths >= 1]
	url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
	url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
	url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
	site_confirmed = requests.get(url_confirmed).content
	site_deaths = requests.get(url_deaths).content
	site_recovered = requests.get(url_recovered).content
	data_confirmed = pd.read_csv(io.StringIO(site_confirmed.decode('utf-8')))
	data_confirmed = data_confirmed.groupby(['Country/Region']).sum()
	data_confirmed = data_confirmed.drop(columns=['Lat','Long'])
	data_deaths = pd.read_csv(io.StringIO(site_deaths.decode('utf-8')))
	data_deaths = data_deaths.groupby(['Country/Region']).sum()
	data_deaths = data_deaths.drop(columns=['Lat','Long'])
	data_recovered = pd.read_csv(io.StringIO(site_recovered.decode('utf-8')))
	data_recovered = data_recovered.groupby(['Country/Region']).sum()
	data_recovered = data_recovered.drop(columns=['Lat','Long'])

series = {}
for country in data.Country.unique():
	series[country] = data[data.Country == country]
	
JH_dataset = {}
for country in data_confirmed.index:
	if not (country in data_confirmed.index
		      and country in data_deaths.index
			  and country in data_recovered.index): continue
	JH_dataset[country] = pd.DataFrame((data_confirmed.loc[country].reset_index(drop=True), 
		                                                   data_deaths.loc[country].reset_index(drop=True),
														   data_recovered.loc[country].reset_index(drop=True)),
	                                                       index = ['Confirmed', 'Deaths', 'Recovered'])
	JH_dataset[country].columns = data_confirmed.loc[country].index
	JH_dataset[country].fillna(value = 0, inplace = True)
	JH_dataset[country] = JH_dataset[country].loc[:, JH_dataset[country].loc['Confirmed']>200]
	
	
CoI = ['Switzerland','Brazil', 'Italy', 'China', 'Japan', 'Korea, South', 'US', 'United Kingdom', 'Spain', 'Germany', 'Iran','Austria']

	
relative_increase = {}
for country in CoI:
	confirmed = np.array(series[country]['Confirmed'])
	relative_increase[country] = (confirmed[1:] - confirmed[:-1]) / confirmed[:-1]
	
relative_deaths = {}
for country in CoI:
	deaths = np.array(series[country]['Deaths'])
	relative_deaths[country] = (deaths[1:] - deaths[:-1]) / deaths[:-1]


if False:
	for c in CoI:
	    print(c, relative_increase[c].mean())

def print_confirmed():	
	for c in CoI:
		print(c)
		plt.plot(moving_average(relative_increase[c]))
		plt.show()
	
def print_deaths():	
	for c in CoI:
		print(c)
		plt.plot(moving_average(relative_deaths[c]))
		plt.show()
		
def print_both():	
	for c in CoI:
		print(c)
		plt.plot(moving_average(relative_deaths[c]))
		plt.plot(moving_average(relative_increase[c]))
		plt.ylim(bottom = 0)
		plt.show()

def print_corrected(extend = False, backtrack = False):
	for c in CoI:
		confirmed = np.array(JH_dataset[c].loc['Confirmed'])
		deaths = np.array(JH_dataset[c].loc['Deaths'])
		recovered = np.array(JH_dataset[c].loc['Recovered'])
		analysis_day = datetime.date.today()
		if backtrack:
			if len(confirmed) - backtrack <= 5: continue
			confirmed = confirmed[:-backtrack]
			deaths = deaths[:-backtrack]
			recovered = recovered[:-backtrack]
			analysis_day = analysis_day - datetime.timedelta(days = backtrack)
		if recovered[-1] == 0: recovered[-1] = recovered[-2]
		if deaths[-1] == 0: deaths[-1] = deaths[-2]
		active = confirmed - (deaths + recovered)
		#safe = deaths+recovered
		#total_increase = confirmed[1:] - confirmed[:-1]
		active_increase = active[1:] - active[:-1]
		#increase = total_increase / (confirmed[:-1] - safe[:-1])
		increase = active_increase / (active[:-1])
		increase = moving_average(increase)
		lin_a, _ = optimize.curve_fit(linear_function, range(len(increase)), increase)[0]
		a, b = optimize.curve_fit(geometric_function, range(len(increase)), increase)[0]
		if b >= 0.999: continue
		vfunc = np.vectorize(geometric_function)
		fit = vfunc(range(len(increase)), a, b)
		if extend:
			recoveries = list(active_increase[-6:])
			last_day = len(confirmed)
			predictions = []
			i = 0
			predictions.append(active[-1] * geometric_function(last_day, a, b) + active[-1])
			recoveries.append(predictions[-1])
			while predictions[-1] > 200:
				i+=1
				predictions.append(predictions[-1] + predictions[-1] * geometric_function(last_day+i, a, b))
				recoveries.append(predictions[-1])
		predictions.insert(0, active[-1] )
		
		fig, (ax1, ax2) = plt.subplots(1, 2)		
		ax1.plot(increase)
		ax1.plot(fit, '--')
		ax1.set_ylim(bottom = -0.5, top = 1.5)
		ax1.set_xlim(left = 0, right = 50)
		
		
		ax2.plot(active)
		ax2.plot( range(last_day-1, last_day+len(predictions)-1), predictions)
		ax2.set_xlim(right = 100)# last_day+len(predictions) )
		ax2.set_ylim(bottom = 100, top = 1000000)
		ax2.set_yscale('log')
		plt.title(c)
		plt.savefig(f'covid-19-master\\{c}_{analysis_day}.jpg')
		plt.show()

def moving_average(arr):
	if arr.size >= 7:
		weight_array = np.array((0.05, 0.25, 0.4,0.5, 0.4, 0.25, 0.05))
	elif arr.size >= 5:
		weight_array = np.array((0.05, 0.25, 0.4, 0.25, 0.05))
	elif arr.size >= 3:
		weight_array = np.array((0.15, 0.4, 0.15))
	else:
		return arr
		
	tail_length = weight_array.size // 2
	summed_weights = np.ones(arr.size)
	summed_weights *= weight_array.sum()
	average = np.copy(arr) * weight_array[weight_array.size//2]
	for i in range(tail_length):
		weight = weight_array[i]
		displacement = tail_length - i
		summed_weights[0 : displacement] -= weight_array[i]
		summed_weights[-1:-displacement-1:-1] -= weight_array[-i+1]
		average[displacement:] += arr[:-displacement] * weight
		average[:-displacement] += arr[displacement:] * weight
	
	return average / summed_weights

def geometric_function(x, a, b):
	return a * b ** x - 0.1

def linear_function(x, a, b):
	return a * x + b
		
print_corrected(extend = True)
#print_corrected(extend = True, backtrack = 1)


for i in range(1,50):
	print_corrected(extend = True, backtrack = i)

		
		
		
		
		
		