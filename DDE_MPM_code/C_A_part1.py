# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:36:06 2020

@author: Yunzhao Ge
"""

import sys,gdal,numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def photo1(points_part,rows,sample_pd_min,sample_pd_max,T):
    #plt.figure(figsize = (24,16))
    plt.figure()
    plt.scatter(points_part[:, 0], points_part[:, 1], color='green', linewidths=0.21)
    plt.ylabel('Log(Area)')
    plt.xlabel('Log(Value)')
    x_major_locator = MultipleLocator(0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    if T:
        text = [['# of Intervals', round(rows / 140)], ['Minimum Value', sample_pd_min[0]],
                ['Maximum Value', sample_pd_max[0]]]
        my_table = plt.table(cellText=text, cellLoc = 'center',colWidths=[0.65]*2,
                     bbox = [0.1, 0.1, 0.65, 0.3])
        my_table.auto_set_font_size(False)
        my_table.set_fontsize(14)
        #my_table.scale(1,4)
    plt.grid(linestyle='-.',which='major')
    plt.savefig(result_output_plot)


#file_path = r'F:\mineral prospecting\C-A\Input\pca1.tif'
#mask_path = r'F:\mineral prospecting\C-A\Input\mask.tif'
#result_output_plot = r'F:\mineral prospecting\C-A\Output\Plot\a.png'

if __name__ == '__main__':
    file_path = sys.argv[1]
    mask_path = sys.argv[2]
    result_output_plot = sys.argv[3]
    result_output_npy = sys.argv[4]

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
    sample_pd.sort_values(by = sample_pd.columns[0],ascending = False, inplace = True)
    sample_np = np.array(sample_pd)

    a = np.log(sample_np)
    b = np.linspace(a.min(), a.max(), 140)
    area = []
    for i in range(140):
        area.append([a[a>=b[i]][-1],np.sum(a>=b[i])])
    area = np.array(area)
    area[:,1] = np.log(area[:,1])
    points_part = area[:-1,:]
    T = True
    np.save(result_output_npy,points_part)
    photo1(points_part,rows,sample_pd_min,sample_pd_max,T)
    print('The successful running')

















