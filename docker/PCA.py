# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:49:43 2020

@author: Yunzhao Ge
"""


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import gdal
from numpy import ma
import math
import sys

def write_geotiff(fname, data, geo_transform, projection,Nodata):

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, eType = gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(Nodata)
    dataset = None

def read_raster_array(file_path):
	value = gdal.Open(file_path)
	value_array = value.GetRasterBand(1).ReadAsArray()
	return value_array

def file_list_array(file_list,mask_path):
	file_list = file_list.split(',')
	file_name = [x.split('/')[-1].split('.')[0].capitalize() for x in file_list]
	file_name.append('Mask')

	file_value = []
	for i in file_list:
		file_value.append(read_raster_array(i))
	file_value.append(read_raster_array(mask_path))

	sample = np.array(file_value)
	dimension = sample.shape
	sample = sample.transpose((1, 2, 0))
	sample = sample.reshape(dimension[1] * dimension[2], dimension[0])
	sample_pd = pd.DataFrame(sample, columns=file_name)
	if len(file_name) - 1 <= 3:
		n_components = len(file_name) - 1
	else:
		n_components = 4
	return sample_pd,dimension,n_components

def data_log(sample_pd,logtrans):
	img = sample_pd.copy()
	sample_pd = sample_pd[sample_pd['Mask'] == 1]
	del sample_pd['Mask']
	indexer_T = img[img['Mask'] == 1].index
	indexer_F = img[img['Mask'] == 0].index
	if logtrans == 'T':
		sample_log = np.log(sample_pd)
		data = sample_log
		img.loc[indexer_T] = np.log(img.loc[indexer_T])
	else:
		data = sample_pd
	u = data.mean()
	σ = data.std()
	data = (data - u) / σ
	img = (img - u) / σ
	return data,img,indexer_T,indexer_F

def pca_value(data,output_plot_value,n_components):
	pca = PCA(copy=True, iterated_power='auto', n_components=n_components,
	          random_state=None, svd_solver='auto', tol=0.0, whiten=False)
	pca.fit(data)
	Component = pca.components_

	rows, columns = data.shape

	NAME = ["PCA1", "PCA2", "PCA3", "PCA4"]
	P = ["A", "B", "C", "D", 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
	Coordinate = list(data.columns)
	#rows = ["Log(%s)" % x for x in Coordinate]


	plt.rcParams['figure.figsize'] = (24, 16)
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 20
	ax1 = plt.subplot(2, 2, 1)
	plt.sca(ax1)
	ind = np.arange(columns)
	P[0] = plt.bar(ind, Component[0, :], width=0.35)
	a = np.zeros((1, columns))
	b = np.r_[a, Component[:n_components, :]]
	for i in range(n_components - 1):
		c = b[:i + 2, :]
		d = np.zeros((2, columns))
		for k in range(columns):
			d[0, k] = sum(c[:, k][c[:, k] < 0])
			d[1, k] = sum(c[:, k][c[:, k] >= 0])
		Top = d[1, :]
		Bottom = d[0, :]
		bottom = []
		for j in range(columns):
			if Component[i + 1][j] > 0:
				bottom.append(Top[j])
			else:
				bottom.append(Bottom[j])
		P[i + 1] = plt.bar(ind, Component[i + 1, :], bottom=bottom, width=0.35)
	plt.ylabel('PCA')
	plt.title('The weight of each element in the PCA')
	plt.xticks(ind, Coordinate)
	plt.legend(P[:n_components], NAME[:n_components], bbox_to_anchor=(1, 1), prop={'size': 12})
	# =================================================================
	ax2 = plt.subplot(2, 2, 2)
	plt.sca(ax2)
	ind = np.arange(n_components)
	P[0] = plt.bar(ind, Component.T[0, :], width=0.35)
	a = np.zeros((1, n_components))
	b = np.r_[a, Component.T]
	for i in range(columns - 1):
		c = b[:i + 2, :]
		d = np.zeros((2, n_components))
		for k in range(n_components):
			d[0, k] = sum(c[:, k][c[:, k] < 0])
			d[1, k] = sum(c[:, k][c[:, k] >= 0])
		Top = d[1, :]
		Bottom = d[0, :]
		bottom = []
		for j in range(n_components):
			if Component.T[i + 1][j] > 0:
				bottom.append(Top[j])
			else:
				bottom.append(Bottom[j])
		P[i + 1] = plt.bar(ind, Component.T[i + 1, :], bottom=bottom, width=0.35)
	plt.title('Eigen Vector')
	plt.xticks(ind, NAME[:n_components])
	plt.legend(P[:columns], Coordinate, bbox_to_anchor=(1, 1), prop={'size': 12})
	# =============================================================
	Component_variance = pca.explained_variance_
	ax3 = plt.subplot(2, 2, 3)
	plt.sca(ax3)
	ind = np.arange(n_components)
	plt.bar(ind, Component_variance, width=0.35)
	plt.ylabel('Variances')
	plt.xlabel('Relative Importance of Principal Components')
	plt.xticks(ind, NAME[:n_components])
	# =============================================================
	ax4 = plt.subplot(2, 2, 4)
	plt.sca(ax4)
	pca = PCA(copy=True, iterated_power='auto', n_components=columns,
	          random_state=None, svd_solver='auto', tol=0.0, whiten=False)
	pca.fit(data)
	Component_variance_ratio = pca.explained_variance_ratio_
	x = np.arange(columns) + 1
	plt.stackplot(x, np.cumsum(Component_variance_ratio))
	plt.ylabel('Cumulative Variance')
	plt.xlabel('Principal Components')
	x_major_locator = MultipleLocator(1)
	ax = plt.gca()
	ax.xaxis.set_major_locator(x_major_locator)
	plt.xlim((1, columns))
	plt.ylim((0, 1))
	plt.savefig(output_plot_value, bbox_inches='tight')

	return pca

def pca_plot(img,pca,indexer_T,indexer_F,dimension,n_components,output_path_npy_list,output_path_plot_pca,output_path_tif_list,mask_path):
    
	del img['Mask']
	pca_value = np.dot(img, pca.components_.T)

	min_value = np.min(pca_value[indexer_T, :], axis=0)
	#max_value = np.max(pca_value[indexer_T, :], axis=0)
	#Nodata = [math.ceil(x - 1) for x in min_value]
	Nodata = min_value - 1
	pca_value[indexer_F, :] = Nodata
	dx, dy = 1, 1
	y, x = np.mgrid[slice(1, dimension[1] + dy, dy), slice(1, dimension[2] + dx, dx)]

	mask_ras = gdal.Open(mask_path)

	plt.close()
	plt.rcParams['figure.figsize'] = (2*dimension[2] / dimension[1] * 24, math.ceil(n_components/2)*24)
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 24

	for i in range(1, n_components + 1):
		pca1 = pca_value[:, i - 1].reshape(dimension[1], dimension[2])
		np.save(output_path_npy_list + '/PCA%s.npy' % i, pca1)

		write_geotiff(output_path_tif_list+'/PCA%s.tif' % i, pca1, mask_ras.GetGeoTransform(),
		              mask_ras.GetProjectionRef(),Nodata[i-1])
		pca1 = pd.DataFrame(pca1)

		pca1_reversed = pca1.iloc[list(reversed(range(dimension[1])))]
		pca1_reversed = ma.masked_where(pca1_reversed < min_value[i - 1], pca1_reversed)

		ax1 = plt.subplot(math.ceil(n_components/2), 2, i)
		plt.sca(ax1)
		cmap = plt.get_cmap('jet')
		cs = plt.pcolormesh(x, y, pca1_reversed, cmap=cmap)
		cbar = plt.colorbar(cs)
		font = {'family': 'serif','color': 'black','weight': 'normal',
		        'size': 24,'rotation': 'vertical'}
		cbar.set_label('PCA%s' % i,fontdict=font)
		plt.xticks([])
		plt.yticks([])
		plt.xlabel('PCA%s' % i, fontsize=38)
	plt.savefig(output_path_plot_pca,bbox_inches='tight')
	print('The successful running')

if __name__ == '__main__':
	file_list = sys.argv[1]
	mask_path = sys.argv[2]
	logtrans = sys.argv[3]
	output_path_plot_value = sys.argv[4]
	output_path_npy_list = sys.argv[5]
	output_path_plot_pca = sys.argv[6]
	output_path_tif_list = sys.argv[7]

	sample_pd, dimension, n_components = file_list_array(file_list, mask_path)
	data, img, indexer_T, indexer_F = data_log(sample_pd, logtrans)
	pca = pca_value(data, output_path_plot_value,n_components)
	pca_plot(img,pca,indexer_T,indexer_F,dimension,n_components,output_path_npy_list,output_path_plot_pca,output_path_tif_list,mask_path)




