# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 17:56:46 2016

@author: Arenhart
"""


import PIL.Image as pil
import PIL.ImageTk as imagetk
import numpy as np
import scipy.ndimage as sp
import matplotlib.pyplot as plt
import skimage as sk
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.measure as measure
import skimage.feature as feature
import tkinter as tk
import tkinter.filedialog as filedialog
import os



MAX_WIDTH = 300
MAX_HEIGHT = 300

def plt_to_image():

	i = 0
	filename = 'temp'+str(i)+'.tif'
	while filename in os.listdir(os.getcwd()):
		i += 1
		filename = 'temp'+str(i)+'.tif'
		if i >= 1000: return None
	plt.savefig(filename)

	with open(filename, 'rb') as file:
		image = pil.open(file)
		img = image.copy()
		image.close()
	os.remove(filename)
	return img


def carregar():
	return pil.open(filedialog.askopenfilename())


def salvar(imagem):
	save_name = filedialog.asksaveasfilename()
	if save_name == '': return
	try:
		imagem.save(save_name)
	except:
		if '.' in save_name:
			save_name = save_name[:save_name.find('.')] + '.bmp'
		else:
			save_name = save_name + '.bmp'
		imagem.save(save_name)


def verificar_binaria(matriz):

	mat = matriz > 0

	return np.sum(mat) * 255 == np.sum(matriz)

def vis(matriz):
	return pil.fromarray(matriz)

def binarizar(matriz, limiar=None):
	if limiar == None:
		limiar = filters.threshold_otsu(matriz)
	return ((matriz >= limiar) *255).astype('uint8')

def histograma(matriz, bins = 254):
	plt.clf()
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
	return (filters.gaussian(matriz,
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
	area_inicial = matriz_binaria.sum()
	menor_aresta = min(matriz_binaria.shape)
	raio = [0]
	area_cf = [0]
	area = [0]
	i = 1
	while area_cf[-1] < 1 and i < menor_aresta and i < 50:
		raio.append(i)
		new_area = 1 - (abertura(matriz_binaria,i).sum()/area_inicial)
		area.append(new_area-area_cf[-1])
		area_cf.append(new_area)
		i += 1
		print(i)
	plt.plot(raio,area,color='blue')
	plt.plot(raio,area_cf,color='green')

def correlacao(matriz_binaria):
	if not matriz_binaria.dtype == 'bool':
		matriz_binaria = (matriz_binaria / matriz_binaria.max()).astype('uint8')
	comprimento = min(matriz_binaria.shape)//2
	correlacao_x = []
	correlacao_y = []
	correlacao_x.append(matriz_binaria.mean())
	for i in range(1,comprimento):
		correlacao_x.append(
				( (matriz_binaria[0:-i,:] * matriz_binaria[i:,:]).sum() )
				 / matriz_binaria[i:,:].size )
		
	correlacao_y.append(matriz_binaria.mean())
	for i in range(1,comprimento):
		correlacao_y.append(
				( (matriz_binaria[:,0:-i] * matriz_binaria[:,i:]).sum() )
				 / matriz_binaria[:,i:].size )
	correlacao_x = np.array(correlacao_x)
	correlacao_y = np.array(correlacao_y)
	correlacao = (correlacao_x + correlacao_y)/2
	plt.plot(range(comprimento),correlacao_x,color='blue')
	plt.plot(range(comprimento),correlacao_y,color='red')
	plt.plot(range(comprimento),correlacao,color='green')
	#plt.show()
	return (correlacao_x, correlacao_y, correlacao)


		
def rotular(imagem_binaria):
	return measure.label(imagem_binaria,background=0)
	
def rotular_colorido(matriz_binaria):
	
	mat_rotulada = measure.label(matriz_binaria,background=0)
	size = matriz_binaria.shape
	mat = np.zeros((size[0],size[1],3),dtype = np.uint8)
	max_index = mat_rotulada.max()
	g_factor = int(max_index**(2/3))
	r_factor = int(max_index**(1/3))
	for i,j in [(i,j) for i in range(size[0]) for j in range(size[1])]:
		index = mat_rotulada[i,j]
		if index == 0:
			mat[i,j,0], mat[i,j,1], mat[i,j,2] = 0, 0, 0
			continue
		b = 50 + int( 205 * (index / max_index) )
		g = 50 + int( (index%g_factor) * (205/g_factor))
		r = 50 + int( (index%r_factor) * (205/r_factor))
		mat[i,j,0], mat[i,j,1], mat[i,j,2] = r,g,b
	
	return mat
  
	
def conectividade(matriz_binaria):
	matriz_binaria = rotular(matriz_binaria)
	comprimento = range(min(matriz_binaria.shape)//2)
	tamanho_total = matriz_binaria.shape[0]*matriz_binaria.shape[1]
	conectividade_x = []
	conectividade_y = []
	matriz = matriz_binaria#.flatten()
	for i in comprimento:
		matriz_deslocada = np.append(matriz[i:,:],matriz[:i,:])
		matriz_sobreposta = np.logical_and(matriz_deslocada==matriz,matriz != -1)
		conectividade_x.append(sum(matriz_sobreposta)/tamanho_total)
	#matriz = matriz_binaria.transpose().flatten()
	for i in comprimento:
		matriz_deslocada = np.append(matriz[:,i:],matriz[:,:i])
		matriz_sobreposta = np.logical_and(matriz_deslocada==matriz,matriz != -1)
		conectividade_y.append(sum(matriz_sobreposta)/tamanho_total)
	conectividade = (np.array(conectividade_x) + np.array(conectividade_y))/2
	plt.plot(comprimento,conectividade_x,color='blue')
	plt.plot(comprimento,conectividade_y,color='red')
	plt.plot(comprimento,conectividade,color='green')
	#plt.show()
		
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
	print ('Areas (media = ' + str(np.mean(areas)) + ' ; desvio padrao = ' + str(np.std(areas))	 + ')')
	plt.hist(areas,bins=bins)
	plt.show()
	print ('Alongamento (media = ' + str(np.mean(alongamento)) + ' ; desvio padrao = ' + str(np.std(alongamento))  + ')')
	plt.hist(alongamento,bins=bins)
	plt.show()
	print ('Rugosidade (media = ' + str(np.mean(rugosidade)) + ' ; desvio padrao = ' + str(np.std(rugosidade))	+ ')')
	plt.hist(rugosidade,bins=bins)
	plt.show()
		
def gerar_ruido_gaussiano(matriz,desv_pad=0.01):
	return (sk.util.random_noise(matriz,var=desv_pad)*255).astype('uint8')

def gerar_ruido_snp(matriz,quantidade=0.1):
	return (sk.util.random_noise(matriz,mode='s&p',amount=quantidade)*255).astype('uint8')

def gerar_imagem_ruido(aresta,densidade):
	return (sk.util.random_noise(np.zeros((aresta[0],aresta[1])),
			mode='salt',amount=densidade)*255).astype('uint8')		   

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
	
	


'''
Interface
'''


class Interface():
	
	def __init__(self, parent):
		
		self.parent = parent
		
		self.img = None
		self.img_desfazer = None
		
		self.main_frame = tk.Frame(self.parent)
		self.main_frame.pack()
		
		self.image_frame = tk.Frame(self.parent)
		self.image_frame.pack(fill=tk.BOTH, expand = 1)
		
		self.canvas = tk.Canvas(self.image_frame,
								relief = tk.SUNKEN)
		self.canvas.config(width=200,height=200)
		self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand = 1)
		
		self.sbV = tk.Scrollbar(self.canvas, orient=tk.VERTICAL)
		self.sbH = tk.Scrollbar(self.canvas, orient=tk.HORIZONTAL)
		
		self.sbV.config(command=self.canvas.yview)
		self.sbH.config(command=self.canvas.xview)
		
		self.canvas.config(yscrollcommand=self.sbV.set)
		self.canvas.config(xscrollcommand=self.sbH.set)
		
		self.sbV.pack(side=tk.RIGHT, fill=tk.Y)
		self.sbH.pack(side=tk.BOTTOM, fill=tk.X)
		
		'''
		Inicializacao dos botoes de menu
		'''
		self.menu = tk.Menu(parent)
		
		
		self.menu_arquivo = tk.Menu(self.menu,tearoff=0)
		self.menu_arquivo.add_command(label="Abrir imagem", 
							  command = self.carregar_imagem)
		self.menu_arquivo.add_command(label="Salvar imagem", 
							  command = self.salvar_imagem)
		self.menu_arquivo.add_command(label="Fechar imagem", 
							  command = self.fechar_imagem)
		self.menu_arquivo.add_command(label="Defazer", 
							  command = self.desfazer)
		self.menu_arquivo.add_command(label="Sair", 
							  command = self.fechar)
		self.menu.add_cascade(label = 'Arquivo', 
							  menu=self.menu_arquivo)
		
		self.menu_transformar = tk.Menu(self.menu,tearoff=0)
		self.menu_transformar.add_command(label='Converter escala de cinza',
										  command = self.escala_de_cinza)
		self.menu_transformar.add_command(label = 'Binarizar...',
										 command = self.binarizar)
		self.menu_transformar.add_command(label = 'Mapa de distancia',
										  command = self.mapa_distancia)
		self.menu_transformar.add_command(label = 'Inverter',
										  command = self.inverter)
		self.menu_transformar.add_command(label = 'Rotular',
										  command = self.rotular)
		self.menu.add_cascade(label="Transformar", 
							  menu = self.menu_transformar)
		
		self.menu_filtros = tk.Menu(self.menu, tearoff = 0)
		self.menu_filtros.add_command(label = 'Expandir Contraste',
									  command = self.expandir_contraste)
		self.menu_filtros.add_command(label = 'Equalizar Histograma',
									  command = self.equalizar_histograma)
		self.menu_filtros.add_command(label = 'Filtro Gaussiano...',
									  command = lambda: self.filtro('init gauss'))
		self.menu_filtros.add_command(label = 'Filtro da Mediana...',
									  command = lambda: self.filtro('init media'))
		self.menu_filtros.add_command(label = 'Filtro Realce...',
									  command = lambda: self.filtro('init real'))
		self.menu_filtros.add_command(label = 'Filtro Prewitt',
									  command = self.filtro_prewitt)
		self.menu_filtros.add_command(label = 'Filtro Sobel',
									  command = self.filtro_sobel)
		self.menu_filtros.add_command(label = 'Filtro Scharr',
									  command = self.filtro_scharr)
		self.menu.add_cascade(label="Filtros", menu = self.menu_filtros)
		
		self.menu_info = tk.Menu(self.menu, tearoff = 0)
		self.menu_info.add_command(label = 'Histograma...',
								   command = self.histograma)
		self.menu_info.add_command(label = 'Correlacao',
								   command = self.correlacao)
		self.menu_info.add_command(label = 'Conectividade',
								   command = self.conectividade)
		self.menu_info.add_command(label = 'Propriedades',
								   command = self.propriedades)
		self.menu.add_cascade(label="Info", menu = self.menu_info)
		
		self.menu_morfologia = tk.Menu(self.menu, tearoff = 0)
		self.menu_morfologia.add_command(label = 'Erosao...',
										 command = lambda: self.morfologia('init erosao'))
		self.menu_morfologia.add_command(label = 'Dilatacao...',
										 command = lambda: self.morfologia('init dilatacao'))
		self.menu_morfologia.add_command(label='Abertura...',
										 command = lambda: self.morfologia('init abertura'))
		self.menu_morfologia.add_command(label = 'Fechamento...',
										 command = lambda: self.morfologia('init fechamento'))
		self.menu_morfologia.add_command(label = 'Granulometria',
										 command = self.granulometria)
		self.menu.add_cascade(label="Morfologia", menu=self.menu_morfologia)
		
		self.menu_ruido = tk.Menu(self.menu, tearoff = 0)
		self.menu_ruido.add_command(label = 'Gerar Ruido Gaussiano...',
									command = lambda: self.filtro('init gaussiano'))
		self.menu_ruido.add_command(label = 'Gerar Ruido "Sal e Pimenta"...',
									command = lambda: self.filtro('init snp'))
		self.menu_ruido.add_command(label = 'Criar Imagem com Ruido...',
									command = lambda: self.gerar_imagem_ruido('init'))
		self.menu.add_cascade(label="Ruido", menu=self.menu_ruido)
				
		self.menu_bordas = tk.Menu(self.menu, tearoff=0)
		self.menu_bordas.add_command(label = 'Extrair Bordas...',
									 command = self.extrair_bordas)
		self.menu_bordas.add_command(label = 'Segregacao Watershed...',
									 command = self.segregacao_watershed)
		self.menu.add_cascade(label="Bordas", menu=self.menu_bordas)
		'''
		fim da inicializacao dos botoes de menu
		'''
		
		
		'''
		Inicializacao janelas secundarias
		'''
		
		# Histograma
		self.janela_histograma = tk.Toplevel(self.parent)
		self.janela_histograma.withdraw()
		self.histograma_show = tk.Label(self.janela_histograma)
		self.histograma_show.pack(side=tk.TOP)
		self.histograma_button = tk.Button(self.janela_histograma,
										   text='Fechar',
										   command = self.janela_histograma.withdraw)
		self.histograma_button.pack(side=tk.TOP)
		
		# Binarizacao
		self.janela_binarizar = tk.Toplevel(self.parent)
		self.janela_binarizar.protocol('WM_DELETE_WINDOW', lambda: print('Invalido'))
		self.janela_binarizar.withdraw()
		self.binarizar_show = tk.Label(self.janela_binarizar)
		self.binarizar_show.pack(side=tk.TOP)
		self.binarizar_botoes = tk.Label(self.janela_binarizar)
		self.binarizar_botoes.pack(side = tk.TOP)
		self.binarizar_fechar = tk.Button(self.binarizar_botoes,
										   text='Cancelar',
										   command = lambda: self.binarizar('cancelar'))		
		self.binarizar_fechar.pack(side=tk.LEFT)
		self.binarizar_ok = tk.Button(self.binarizar_botoes,
									  text = 'OK',
									  command = lambda: self.binarizar('confirmar'))
		self.binarizar_ok.pack(side = tk.LEFT)
		self.binarizar_parametros = tk.Label(self.janela_binarizar)
		self.binarizar_parametros.pack(side = tk.TOP)
		self.label_limiar = tk.Label(self.binarizar_parametros,
									 text = 'Limiar()')
		self.label_limiar.grid(row=0,column=0)
		self.limiar_binarizacao = tk.StringVar()
		self.entry_limiar = tk.Entry(self.binarizar_parametros,
									 textvariable = self.limiar_binarizacao)
		self.entry_limiar.grid(row=0,column=1)
		self.limiar_binarizacao.trace('w',lambda a,b,c:	 self.binarizar('atualizar'))
		self.binarizar_botao_aumentar = tk.Button(self.binarizar_parametros,
												  text = '+',
												  command = lambda: self.binarizar('aumentar'))
		self.binarizar_botao_aumentar.grid(row=0,column=2)
		self.binarizar_botao_diminuir = tk.Button(self.binarizar_parametros,
												  text = '-',
												  command = lambda: self.binarizar('diminuir'))
		self.binarizar_botao_diminuir.grid(row=0,column=3)
		
		# Filtros
		self.funcao_filtro = None
		self.janela_filtro = tk.Toplevel(self.parent)
		self.janela_filtro.protocol('WM_DELETE_WINDOW', lambda: print('Invalido'))
		self.filtro_label = tk.Label(self.janela_filtro)
		self.filtro_label.grid(row = 0, column = 0)
		self.filtro_var = tk.StringVar()
		self.filtro_var.trace('w', lambda a,b,c: self.funcao_filtro('atualizar'))
		self.filtro_campo = tk.Entry(self.janela_filtro,
									 textvariable = self.filtro_var)
		self.filtro_campo.grid(row = 0, column = 1)
		self.filtro_botao_aumentar = tk.Button(self.janela_filtro,
											   text = '+',
											   command = lambda: self.funcao_filtro('aumentar'))
		self.filtro_botao_aumentar.grid(row=0, column = 2)
		self.filtro_botao_diminuir = tk.Button(self.janela_filtro,
											   text = '-',
											   command = lambda: self.funcao_filtro('diminuir'))
		self.filtro_botao_diminuir.grid(row=0, column = 3)
		self.filtro_botao_ok = tk.Button(self.janela_filtro,
										 text = 'OK',
										 command = lambda: self.funcao_filtro('aceitar'))
		self.filtro_botao_ok.grid(row=1, column = 0)
		self.filtro_botao_cancelar = tk.Button(self.janela_filtro,
											   text = 'Cancelar',
											   command = lambda: self.funcao_filtro('cancelar'))
		self.filtro_botao_cancelar.grid(row=1, column = 1)
		self.janela_filtro.withdraw()
		
		# Ruido
		self.janela_ruido = tk.Toplevel(self.parent)
		self.ruido_var1 = tk.StringVar()
		self.ruido_var1.set('100')
		self.ruido_var1.trace('w', lambda a,b,c: self.gerar_imagem_ruido('atualizar'))
		self.ruido_label1 = tk.Label(self.janela_ruido, text = 'Altura(100): ')
		self.ruido_label1.grid(column = 0, row = 0)
		self.ruido_entry1 = tk.Entry(self.janela_ruido, 
									 textvariable = self.ruido_var1)
		self.ruido_entry1.grid(row = 0, column = 1)
		self.ruido_var2 = tk.StringVar()
		self.ruido_var2.set('100')
		self.ruido_var2.trace('w', lambda a,b,c: self.gerar_imagem_ruido('atualizar'))
		self.ruido_label2 = tk.Label(self.janela_ruido, text = 'Largura(100): ')
		self.ruido_label2.grid(column = 0, row = 1)
		self.ruido_entry2 = tk.Entry(self.janela_ruido, 
									 textvariable = self.ruido_var2)
		self.ruido_entry2.grid(column=1, row=1)
		self.ruido_var3 = tk.StringVar()
		self.ruido_var3.set('0.5')
		self.ruido_var3.trace('w', lambda a,b,c: self.gerar_imagem_ruido('atualizar'))
		self.ruido_label3 = tk.Label(self.janela_ruido, text = 'Proporcao(0.5): ')
		self.ruido_label3.grid(column = 0, row = 2)
		self.ruido_entry3 = tk.Entry(self.janela_ruido, 
									 textvariable = self.ruido_var3)
		self.ruido_entry3.grid(column=1, row=2)
		
		self.ruido_ok = tk.Button(self.janela_ruido,
								  text = 'OK',
								  command = lambda: self.gerar_imagem_ruido('aceitar'))
		self.ruido_ok.grid(column = 0, row = 3)
		self.ruido_cancelar = tk.Button(self.janela_ruido,
										text = 'Cancelar',
										command = lambda: self.gerar_imagem_ruido('cancelar'))
		self.ruido_cancelar.grid(row = 3, column = 1)
		
		self.janela_ruido.withdraw()
		
		
		
		parent.config(menu=self.menu)
		parent.geometry('400x300')
	
	def salvar_imagem(self):
		if self.img != None:
			salvar(self.img)
		
	def fechar(self):
		self.parent.quit()
		self.parent.destroy()
		
	def carregar_imagem(self):
		
		self.img = pil.open(filedialog.askopenfilename())
		self.img_desfazer = None
		
		self.atualizar()
	
	def fechar_imagem(self):
		self.img, self.img_desfazer = None, self.img
		self.atualizar()
		
	def desfazer(self):
		
		if self.img_desfazer == None:
			print ('Sem imagem para desfazer')
			return
		
		self.img, self.img_desfazer = self.img_desfazer, self.img
		self.atualizar()
		
	def atualizar(self):
		
		self.canvas.delete('all')
		
		if self.img == None:
			return
			
		self.photo_img = imagetk.PhotoImage(self.img)
		size = self.img.size
		self.canvas_image = self.canvas.create_image(0,0,anchor='nw', 
													 image=self.photo_img)
		self.canvas.config(width=min(size[0],MAX_WIDTH),
							height=min(size[1],MAX_HEIGHT) )
		self.canvas.config(scrollregion=(0,0,size[0],size[1]))

	def escala_de_cinza(self):
		
		if self.img.mode == 'L':
			return		  
		self.img_desfazer = self.img
		self.img = self.img.convert('L')
		self.atualizar()

	def binarizar(self, modo = 'iniciar'):
		'''
		Modos: iniciar, confirmar, atualizar, cancelar, aumentar, diminuir
		'''
		
		if modo == 'iniciar':
			
			if self.img.mode == 'L':
				mat = np.array(self.img)
			else:
				mat = np.array(self.img.convert('L'))
			
			histograma(mat) #grava o grafico para uma variavel interna, plt.gcp()
			self.hist_img = plt_to_image().copy()
			self.hist_img = imagetk.PhotoImage(self.hist_img.convert('RGB'))
			self.hist_ref = self.hist_img
			self.binarizar_show.config(image=self.hist_img)
	
			
			self.janela_binarizar.deiconify()
			self.otsu = filters.threshold_otsu(np.array(self.img))
			self.label_limiar.configure(text = 'Limiar ('+str(self.otsu)+') :')			   
			self.img_original = self.img.copy()
			self.limiar_valido = self.otsu
			self.limiar_binarizacao.set(self.otsu)
			
		elif modo == 'confirmar':
			self.img_desfazer = self.img_original
			self.janela_binarizar.withdraw()
			
		elif modo == 'atualizar':
			
			if not self.limiar_binarizacao.get().isdigit() and self.limiar_binarizacao.get() != '':
				self.limiar_binarizacao.set(self.limiar_valido)
				return
			elif self.limiar_binarizacao.get() == "":
				self.limiar_valido = ""
				return

			self.limiar_valido = int(self.limiar_binarizacao.get())
			self.img = binarizar(np.array(self.img_original),
								 int(self.limiar_binarizacao.get()))
			self.img = pil.fromarray(self.img)
			self.atualizar()
			
		elif modo == 'cancelar':
			self.img = self.img_original
			self.atualizar()
			self.janela_binarizar.withdraw()
			
		elif modo == 'aumentar':
			self.limiar_binarizacao.set(str(int(self.limiar_binarizacao.get())+1))
			
		elif modo == 'diminuir':
			self.limiar_binarizacao.set(str(int(self.limiar_binarizacao.get())-1))


	def histograma(self):
		
		if self.img.mode == 'L':
			mat = np.array(self.img)
		else:
			mat = np.array(self.img.convert('L'))
		histograma(mat) #grava o grafico para uma variavel interna, plt.gcp()
		self.hist_img = plt_to_image().copy()
		self.hist_img = imagetk.PhotoImage(self.hist_img.convert('RGB'))
		self.hist_ref = self.hist_img
		self.histograma_show.config(image=self.hist_img)

		self.janela_histograma.deiconify()
		
	
	def mapa_distancia(self):
		mat = np.array(self.img)
		binaria = verificar_binaria(mat)
		print (binaria)
		if not binaria:
			return
		
		self.img_desfazer = self.img.copy()
		self.img = pil.fromarray(mapa_distancia(mat))
		self.atualizar()
	
	def inverter(self):
		if self.img.mode == 'L':
			mat = np.array(self.img)
			mat = inverter(mat)
		else:
			mat1, mat2, mat3 = self.img.split()
			mat1, mat2, mat3 = list(map(np.array,(mat1,mat2,mat3)))
			mat1, mat2, mat3 = list(map(inverter,(mat1,mat2,mat3)))
			mat = np.stack((mat1, mat2, mat3),axis=-1)
		self.img_desfazer = self.img	
		self.img = pil.fromarray(mat)
		self.atualizar()
	
	def expandir_contraste(self):
		mat = np.array(self.img)
		
		self.img_desfazer = self.img.copy()
		self.img = pil.fromarray(expandir_contraste(mat))
		self.atualizar()
	
	def equalizar_histograma(self):
		mat = np.array(self.img)
		
		self.img_desfazer = self.img.copy()
		self.img = pil.fromarray(equalizar_histograma(mat))
		self.atualizar()
	
	def filtro(self,modo):
		'''
		modos: init gauss, init media, init real, atualizar, confirmar, 
				cancelar, aumentar, diminuir
		'''
		print(modo)
		if modo.split()[0] == 'init':
			
			self.funcao_filtro = self.filtro 
			filtro = modo.split()[1]
			if	filtro == 'gauss':
				self.filtro_atual = filtro_gaussiano
				self.janela_filtro.title('Filtro Gaussiano')
				self.filtro_label.config(text = 'Desvio Padrao (1.0)')
			elif filtro == 'media':
				if self.img.mode != 'L':
					print ('Filtro disponivel apenas em escala de cinza')
					return
				self.filtro_atual = filtro_mediana
				self.janela_filtro.title('Filtro da Mediana')
				self.filtro_label.config(text = 'Tamanho Disco (1.0)')
			elif filtro == 'real':
				if self.img.mode != 'L':
					print ('Filtro disponivel apenas em escala de cinza')
					return
				self.filtro_atual = filtro_realce
				self.janela_filtro.title('Filtro da Realce')
				self.filtro_label.config(text = 'Tamanho Disco (1.0)')
				
			elif filtro == 'gaussiano':
				self.filtro_atual = gerar_ruido_gaussiano
				self.janela_filtro.title('Filtro Gaussiano')
				self.filtro_label.config(text = 'Desvio Padrao (0.1)')
 
			elif filtro == 'snp':
				self.filtro_atual = gerar_ruido_snp
				self.janela_filtro.title('Filtro Sal e Pimenta')
				self.filtro_label.config(text = 'Densidade (0.1)')
				
			self.janela_filtro.deiconify()
			self.img_original = self.img.copy()	   
			self.filtro_var.set('1.0')
			self.filtro_var_valida = '1.0'
			
		elif modo == 'atualizar':
			
			if self.filtro_var.get() == '':
				self.filtro_var_valida = ''
				self.img = self.img_original
				self.atualizar()
				return
				
			valid = [i for i in self.filtro_var.get() 
					   if  i in '1234567890.']
			if len(valid) != len(self.filtro_var.get()) or self.filtro_var.get().count('.') > 1:
				self.filtro_var.set(self.filtro_var_valida)
				return
			
			print ('atualizando')
			mat = np.array(self.img_original)
			mat = self.filtro_atual(mat,float(self.filtro_var.get()))
			self.img = pil.fromarray(mat)
			self.atualizar()
			
		
		elif modo == 'cancelar':
			self.img = self.img_original
			self.janela_filtro.withdraw()
			self.atualizar()
			
		elif modo == 'aceitar':
			self.img_defazer = self.img_original
			self.janela_filtro.withdraw()
			
		elif modo == 'aumentar':
			self.filtro_var.set(str(float(self.filtro_var.get())+1))
			
		elif modo == 'diminuir':
			if float(self.filtro_var.get()) < 1:
				self.filtro_var.set('0')
			else:
				self.filtro_var.set(str(float(self.filtro_var.get())-1))
		
	
	def filtro_prewitt(self):
		if self.img.mode == 'L':
			mat = np.array(self.img)
			mat = filtro_prewitt(mat)
		else:
			mat1, mat2, mat3 = self.img.split()
			mat1, mat2, mat3 = list(map(np.array,(mat1,mat2,mat3)))
			mat1, mat2, mat3 = list(map(filtro_prewitt,(mat1,mat2,mat3)))
			mat = (mat1 + mat2 + mat3)//3
		self.img_desfazer = self.img	
		self.img = pil.fromarray(mat)
		self.atualizar()
	
	def filtro_sobel(self):
		if self.img.mode == 'L':
			mat = np.array(self.img)
			mat = filtro_sobel(mat)
		else:
			mat1, mat2, mat3 = self.img.split()
			mat1, mat2, mat3 = list(map(np.array,(mat1,mat2,mat3)))
			mat1, mat2, mat3 = list(map(filtro_sobel,(mat1,mat2,mat3)))
			mat = (mat1 + mat2 + mat3)//3
		self.img_desfazer = self.img	
		self.img = pil.fromarray(mat)
		self.atualizar()
	
	def filtro_scharr(self):
		if self.img.mode == 'L':
			mat = np.array(self.img)
			mat = filtro_scharr(mat)
		else:
			mat1, mat2, mat3 = self.img.split()
			mat1, mat2, mat3 = list(map(np.array,(mat1,mat2,mat3)))
			mat1, mat2, mat3 = list(map(filtro_scharr,(mat1,mat2,mat3)))
			mat = (mat1 + mat2 + mat3)//3
		self.img_desfazer = self.img	
		self.img = pil.fromarray(mat)
		self.atualizar()
	
		
	def morfologia(self, modo):
		'''
		modos: init erosao, init dilatacao, init abertura, init fechamento,
		atualizar, cancelar, confirmar, aumentar, diminuir
		'''
		
		binario = verificar_binaria(np.array(self.img))
		if not binario:
			print ('funcao apenas para imagens binarizadas')
			return
		
		
		if modo.split()[0] == 'init':
			self.funcao_filtro = self.filtro
			filtro = modo.split()[1]
			if	filtro == 'erosao':
				self.filtro_atual = erosao
				self.janela_filtro.title('Erosao')
				self.filtro_label.config(text = 'Erosao(1): ')
				
			elif filtro == 'dilatacao':
				self.filtro_atual = dilatacao
				self.janela_filtro.title('Dilatacao')
				self.filtro_label.config(text = 'Dilatacao(1): ')
				
			elif filtro == 'abertura':
				self.filtro_atual = abertura
				self.janela_filtro.title('Abertura')
				self.filtro_label.config(text = 'Abertura(1): ')
				
			elif filtro == 'fechamento':
				self.filtro_atual = fechamento
				self.janela_filtro.title('Fechamento')
				self.filtro_label.config(text = 'Fechamento (1): ')
				
			self.janela_filtro.deiconify()
			self.img_original = self.img.copy()	   
			self.filtro_var.set('1')
			self.filtro_var_valida = '1'
			
		elif modo == 'atualizar':
			
			if self.filtro_var.get() == '' or '0':
				self.filtro_var_valida = ''
				self.img = self.img_original
				self.atualizar()
				return
				
			valid = [i for i in self.filtro_var.get() 
					   if  i in '1234567890']
			if len(valid) != len(self.filtro_var.get()) or self.filtro_var.get().count('.') > 1:
				self.filtro_var.set(self.filtro_var_valida)
				return
			
			mat = np.array(self.img_original)
			mat = self.filtro_atual(mat,float(self.filtro_var.get()))
			self.img = pil.fromarray(mat)
			self.atualizar()
			
		
		elif modo == 'cancelar':
			self.img = self.img_original
			self.janela_filtro.withdraw()
			self.atualizar()
			
		elif modo == 'aceitar':
			self.img_defazer = self.img_original
			self.janela_filtro.withdraw()
			
		elif modo == 'aumentar':
			self.filtro_var.set(str(float(self.filtro_var.get())+1))
			
		elif modo == 'diminuir':
			if float(self.filtro_var.get()) < 1:
				self.filtro_var.set('0')
			else:
				self.filtro_var.set(str(float(self.filtro_var.get())-1))
		
	
	def granulometria(self):
		binario = verificar_binaria(np.array(self.img))
		if not binario:
			print ('funcao apenas para imagens binarizadas')
			return
		
		plt.clf()
		mat = np.array(self.img)
		granulometria(mat)
		self.granulometria_img = plt_to_image().copy()
		self.granulometria_img = imagetk.PhotoImage(
										self.granulometria_img.convert('RGB'))
		self.gran_img_ref = self.granulometria_img
		self.histograma_show.config(image=self.granulometria_img)

		self.janela_histograma.deiconify()
	
	def correlacao(self):
		binario = verificar_binaria(np.array(self.img))
		if not binario:
			print ('funcao apenas para imagens binarizadas')
			return
		
		plt.clf()
		mat = np.array(self.img)
		correlacao(mat)
		self.granulometria_img = plt_to_image().copy()
		self.granulometria_img = imagetk.PhotoImage(
										self.granulometria_img.convert('RGB'))
		self.gran_img_ref = self.granulometria_img
		self.histograma_show.config(image=self.granulometria_img)

		self.janela_histograma.deiconify()
	
	def rotular(self):
		
		binaria =verificar_binaria(np.array(self.img))
		if not binaria:
			print('Funcao apenas para imagens binarias')
			return
		
		mat = np.array(self.img)
		
		self.img_desfazer = self.img.copy()
		mat = rotular_colorido(mat)
		print(mat.max())
		self.img = pil.fromarray(mat)
		self.atualizar()
	
	def conectividade(self):
		binario = verificar_binaria(np.array(self.img))
		if not binario:
			print ('funcao apenas para imagens binarizadas')
			return
		
		plt.clf()
		mat = np.array(self.img)
		conectividade(mat)
		self.granulometria_img = plt_to_image().copy()
		self.granulometria_img = imagetk.PhotoImage(
										self.granulometria_img.convert('RGB'))
		self.gran_img_ref = self.granulometria_img
		self.histograma_show.config(image=self.granulometria_img)

		self.janela_histograma.deiconify()
	
	def propriedades(self):
		pass
	
	def gerar_ruido(self,modo):

		
		if modo == 'init gaussiano':
			self.funcao_filtro = self.gerar_ruido
			self.filtro_atual = gerar_ruido_gaussiano
			self.janela_filtro.title('Filtro Gaussiano')
			self.filtro_label.config(text = 'Desvio Padrao (1.0)')
			self.janela_filtro.deiconify()
			self.img_original = self.img.copy()	   
			self.filtro_var.set('1.0')
			self.filtro_var_valida = '1.0'
			
		elif modo == 'init snp':
			self.funcao_filtro = self.gerar_ruido
			self.filtro_atual = gerar_ruido_snp
			self.janela_filtro.title('Filtro Sal e Pimenta')
			self.filtro_label.config(text = 'Densidade (0.1)')
			self.janela_filtro.deiconify()
			self.img_original = self.img.copy()	   
			self.filtro_var.set('0.1')
			self.filtro_var_valida = '0.1'
	
	def gerar_ruido_snp(self):
		pass
	
	def gerar_imagem_ruido(self, modo):
		'''
		modos: init, atualizar, aceitar, cancelar
		'''
		if not self.img == None:
			print('Gerar imagens apenas se nao houver imagem carregada')
			return
		
		if modo == 'init':
			self.ruido_var1_valida = self.ruido_var1.get()
			self.ruido_var2_valida = self.ruido_var2.get()
			self.ruido_var3_valida = self.ruido_var3.get()
			self.janela_ruido.deiconify()

			
		elif modo == 'atualizar':
			for var in (self.ruido_var1,):
				valid = [i for i in var.get() 
					   if  i in '1234567890']
				if len(valid) != len(var.get()) or var.get().count('.') > 1:
					var.set(self.ruido_var1_valida)
					return
			
			for var in (self.ruido_var2,):
				valid = [i for i in var.get() 
					   if  i in '1234567890']
				if len(valid) != len(var.get()) or var.get().count('.') > 1:
					var.set(self.ruido_var2_valida)
					return
				
			for var in (self.ruido_var3,):
				valid = [i for i in var.get() 
					   if  i in '1234567890.']
				if len(valid) != len(var.get()) or var.get().count('.') > 1:
					var.set(self.ruido_var3_valida)
					return
			self.ruido_var1_valida = self.ruido_var1.get()
			self.ruido_var2_valida = self.ruido_var2.get()
			self.ruido_var3_valida = self.ruido_var3.get()
	
		elif modo == 'cancelar':
			self.janela_ruido.withdraw()
			
		elif modo == 'aceitar':
			largura = int(self.ruido_var1.get())
			altura = int(self.ruido_var2.get())
			densidade = float(self.ruido_var3.get())
			self.img = pil.fromarray(gerar_imagem_ruido(
									(largura,altura),densidade))
			self.atualizar()
			self.janela_ruido.withdraw()
			
	def extrair_bordas(self):
		
		
		binaria = verificar_binaria(np.array(self.img))
		if not binaria:
			print('Funcao apenas para imagens binarias')
			return
		
		mat = np.array(self.img)
		
		self.img_desfazer = self.img.copy()
		mat = extrair_bordas(mat, mediana = 1, gaussiano = 2, realce = 2, 
				   limiar = None, mediana2 = 0,
				   janela = 100, offset = 0)
		print(mat.max())
		self.img = pil.fromarray(mat)
		self.atualizar()
	
	def segregacao_watershed(self):
		binaria = verificar_binaria(np.array(self.img))
		if not binaria:
			print('Funcao apenas para imagens binarias')
			return
		
		mat = np.array(self.img)
		
		self.img_desfazer = self.img.copy()
		mat = segregacao_watershed(mat, pegada = 5, limiar = 0)
		print(mat.max())
		self.img = pil.fromarray(mat)
		self.atualizar()
		
		
		
root = tk.Tk()
root.title('Imaginos')
interface = Interface(root)
root.mainloop()
	
	
	
	
	