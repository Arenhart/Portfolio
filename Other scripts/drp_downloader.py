# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:27:45 2020

@author: Rafael Arenhart
"""

from bs4 import BeautifulSoup
import requests
import re
import os
import time

ROOT = "https://www.digitalrocksportal.org/"

print("Digital Rock Portal Tiff downloader\n")
base_url = input("Enter image repository page: ")
start_page = int(input("Enter starting page: "))
end_page = int(input("Enter last page: "))
sample_name = input("Choose image name to save: ")

if not sample_name in os.listdir():
	os.mkdir(sample_name)
	
image_index = (start_page-1) * 20
for i in range(start_page, end_page + 1):
	print(f'Starting page {i}')

	url = base_url + f"?page={i}"
	time.sleep(2)
	r = requests.get(url)
	soup = BeautifulSoup(r.text, 'xml')
	samples = soup.find_all('ul', class_ = 'dropdown-menu')
	for sample in samples:
		image_index += 1
		print(f'Downloading image {image_index}')
		download_link = sample.find(href=re.compile('/download/'))['href']
		image_url = ROOT[:-1] + download_link
		file_name = sample_name + f'/{sample_name}_{image_index:04.0f}.tif'
		
		if file_name.split('/')[1] in os.listdir(os.getcwd() + '\\' + sample_name):
			continue
		
		while True:	
			try:
				time.sleep(2)
				page = requests.get(image_url, timeout = 10)
				with open(file_name, 'wb') as file:
					file.write(page.content)
				break
			except requests.exceptions.Timeout:
				print(f"Retrying {image_index}")
			except requests.ConnectionError:
				print(f"Retrying {image_index}")