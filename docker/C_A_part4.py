# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:38:54 2020

@author: Yunzhao Ge
"""

import sys,gdal,numpy as np,pandas as pd
import matplotlib.pyplot as plt
from numpy import ma


def phote3(x,y,img_reversed,levels):
    plt.figure()
    cmap = plt.get_cmap('jet')
    cs = plt.contourf(x,y,img_reversed, levels,cmap = cmap)
    cbar = plt.colorbar(cs)
#plt.title('PCA1',fontsize = 28)
    font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        'rotation':'vertical'}
    cbar.set_label('C-A_values',fontdict=font)
#plt.text(round(dimension[2]/2),-8,'PCA1',fontsize = 38)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('C-A',fontsize = 38)
#plt.savefig(r'C:\Users\Administrator\Desktop\PCAA.png',bbox_inches='tight',dpi = 72)
    plt.savefig(result_output_plot)


def write_geotiff(fname, data, geo_transform, projection,Nodata = 0):

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, eType = gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(Nodata)
    dataset = None



if __name__ == '__main__':
    

    file_path = sys.argv[1]
    mask_path = sys.argv[2]
    csv_path  = sys.argv[3]
    result_output_plot = sys.argv[4]
    result_output_npy = sys.argv[5]
    result_output_tif = sys.argv[6]


    file_name = []
    file_value = []
    file_name.append(file_path.split('/')[-1].split('.')[0].capitalize())
    file_name.append('Mask')
    value = gdal.Open(file_path)
    file_value.append(value.GetRasterBand(1).ReadAsArray())
    mask = gdal.Open(mask_path)
    file_value.append(mask.GetRasterBand(1).ReadAsArray()) 
    sample = np.array(file_value)
    dimension = sample.shape
    sample = sample.transpose((1,2,0))
    sample = sample.reshape(dimension[1]*dimension[2],dimension[0])
    sample_pd = pd.DataFrame(sample,columns = file_name)
    img = sample_pd.copy()
    sample_pd = sample_pd[sample_pd['Mask']==1]
    del sample_pd['Mask']
    indexer_T = img[img['Mask']==1].index
    indexer_F = img[img['Mask']==0].index
    del img['Mask']
    rows,columns = sample_pd.shape
    sample_pd_min = sample_pd.min()
    sample_pd_max = sample_pd.max()
    sample_pd = sample_pd - sample_pd_min+0.1
    img = img-sample_pd_min+0.1
    


    min_value = img.iloc[indexer_T].min()[0]
    max_value = img.iloc[indexer_T].max()[0]
    Nodata = min_value - 1
    img.iloc[indexer_F] = Nodata
    img = np.array(img).reshape(dimension[1],dimension[2])
    img = pd.DataFrame(img)
    img_reversed = img.iloc[list(reversed(range(dimension[1])))]
    img_reversed = ma.masked_where(img_reversed < min_value, img_reversed)
    
    
    text_pd = pd.read_csv(csv_path)
    text_pd = text_pd.loc['1':'3','0':'5']
    levels = [float(i) for i in text_pd['2']]
    levels.insert(0,min_value)
    levels.append(max_value)

    c_a_mask = np.zeros(img.shape)
    for i in levels:
        c = img.copy()
        c_a_mask += np.where(c > i, 1, 0)
    np.save(result_output_npy, c_a_mask)

    write_geotiff(result_output_tif , c_a_mask, mask.GetGeoTransform(),
                  mask.GetProjectionRef())

    dx, dy = 1, 1
    y, x = np.mgrid[slice(1, dimension[1] + dy, dy),slice(1, dimension[2] + dx, dx)]
    plt.rcParams['figure.figsize'] = (dimension[2]/dimension[1]*24,24)
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24
    phote3(x,y,img_reversed,levels)  
    print('The successful running')





