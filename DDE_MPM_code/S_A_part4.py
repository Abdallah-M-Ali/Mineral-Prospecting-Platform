# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:46:46 2020

@author: LGSYRZ
"""

import sys,gdal,numpy as np,pandas as pd
import matplotlib.pyplot as plt





def photo3(x,y,iimg_reversed,name,path):
    plt.figure()
    cmap = plt.get_cmap('jet')
    cs = plt.pcolormesh(x, y, iimg_reversed, cmap=cmap)
    cbar = plt.colorbar(cs)
    #plt.title('S-A-%s' % name, fontsize=28)
    font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 24, 'rotation': 'vertical'}
    cbar.set_label('S-A-%s' % name, fontdict=font)
    # plt.text(round(dimension[2]/2),-8,'PCA1',fontsize = 38)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('S-A-%s' % name, fontsize=38)
    #plt.show()
    plt.savefig(path)

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
    csv_path = sys.argv[3]
    result_output_plot_ano = sys.argv[4]
    result_output_plot_back = sys.argv[5]
    result_output_npy_ano = sys.argv[6]
    result_output_npy_back = sys.argv[7]
    result_output_tif_ano = sys.argv[8]
    result_output_tif_back = sys.argv[9]

    value = gdal.Open(file_path)
    img = value.GetRasterBand(1).ReadAsArray()
    mask = gdal.Open(mask_path)
    img_mask = mask.GetRasterBand(1).ReadAsArray()
    img_mask = np.where(img_mask==0,False,True)
    img[~img_mask] = img[img_mask].mean()

    f = np.fft.fft2(img)

    dimension = img_mask.shape
    dx, dy = 1, 1
    y, x = np.mgrid[slice(1, dimension[0] + dy, dy), slice(1, dimension[1] + dx, dx)]
    fig_name = ['Abnormity','Background']
    text_pd = pd.read_csv(csv_path)
    threshold_value = float(np.array(text_pd)[-2][3])
    plt.rcParams['figure.figsize'] = (dimension[1] / dimension[0] * 24, 24)
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24
    for i in fig_name:
        f1 = f.copy()
        if i =='Abnormity':
            f1[f1.real>threshold_value**0.5]=0
            iimg = np.fft.ifft2(f1)
            iimg = iimg.real
            iimg[~img_mask] = np.nan
            np.save(result_output_npy_ano,iimg)
            write_geotiff(result_output_tif_ano, iimg, mask.GetGeoTransform(),
                          mask.GetProjectionRef())

            iimg = pd.DataFrame(iimg)
            iimg_reversed = iimg.iloc[list(reversed(range(dimension[0])))]
            photo3(x,y,iimg_reversed,i,result_output_plot_ano)
        elif i =='Background':
            f1[f1.real < threshold_value ** 0.5] = 0
            iimg = np.fft.ifft2(f1)
            iimg = iimg.real
            iimg[~img_mask] = np.nan
            np.save(result_output_npy_back,iimg)
            write_geotiff(result_output_tif_back, iimg, mask.GetGeoTransform(),
                          mask.GetProjectionRef())
            iimg = pd.DataFrame(iimg)
            iimg_reversed = iimg.iloc[list(reversed(range(dimension[0])))]
            photo3(x,y,iimg_reversed,i,result_output_plot_back)
    print('The successful running')
