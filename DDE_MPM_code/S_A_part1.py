# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:18:31 2020

@author: LGSYRZ
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import gdal
import sys



def photo2(points_part,T,rows,sample_pd_min,sample_pd_max):
    plt.figure()
    plt.scatter(points_part[:, 0], points_part[:, 1], color='green', linewidths=0.21)
    plt.ylabel('Log(Area)')
    plt.xlabel('Log(Value)')
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    if T:
        text = [['# of Intervals',round(rows / 140)],['Minimum Value',sample_pd_min[0]],['Maximum Value',sample_pd_max[0]]]
        my_table = plt.table(cellText=text, cellLoc='center', colWidths=[0.6]*2
                         ,bbox=[0.1, 0.1, 0.7, 0.3])
        my_table.auto_set_font_size(False)
        my_table.set_fontsize(14)
        #my_table.scale(1, 4)
    plt.grid(linestyle='-.', which='major')
    plt.savefig(result_output_plot)





if __name__ == '__main__':
    file_path = sys.argv[1]
    mask_path = sys.argv[2]
    result_output_plot = sys.argv[3]
    result_output_npy = sys.argv[4]
    
    value = gdal.Open(file_path)
    img = value.GetRasterBand(1).ReadAsArray()
    value = gdal.Open(mask_path)
    img_mask = value.GetRasterBand(1).ReadAsArray()
    img_mask = np.where(img_mask==0,False,True)
    img[~img_mask] = img[img_mask].mean()

    f = np.fft.fft2(img)
    S = f.real**2

    Mask_pd = pd.DataFrame(img_mask.reshape(1,-1).T)
    sample_pd = pd.DataFrame(S.reshape(1,-1).T,columns = ['value'])
    indexer_T = Mask_pd[Mask_pd[0]==True].index
#indexer_F = Mask_pd[Mask_pd[0]==False].index
    sample_pd = sample_pd.iloc[indexer_T]
    sample_pd.sort_values(by = ['value'],ascending = False, inplace = True)
    sample_np = np.array(sample_pd)
    sample_pd_min = sample_pd.min()
    sample_pd_max = sample_pd.max()
    rows,columns = sample_pd.shape
    a = np.log(sample_np)
    interval = 140
    b = np.linspace(a.min(), a.max(), interval)
    area = []
    for i in range(interval):
        area.append([a[a>=b[i]][-1],np.sum(a>=b[i])])
    area = np.array(area)
    area[:,1] = np.log(area[:,1])
    points_part = area[:-1,:]
    
    np.save(result_output_npy,points_part)
    T = True
    photo2(points_part,T,rows,sample_pd_min,sample_pd_max)
    print('The successful running')











