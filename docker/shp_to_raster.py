# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 01:14:58 2020

@author: Yunzhao Ge
"""

import gdal,ogr,sys


def shp_raster(shpFile,rasterFile,outraster):

    dataset = gdal.Open(rasterFile)
    geo_transform = dataset.GetGeoTransform()  
    cols = dataset.RasterXSize    
    rows = dataset.RasterYSize    
    shp = ogr.Open(shpFile,0)
    m_layer = shp.GetLayerByIndex(0)
    target_ds = gdal.GetDriverByName('GTiff').Create(outraster, xsize=cols, ysize=rows, bands=1,
    eType = gdal.GDT_Float32)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(dataset.GetProjection())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], m_layer,burn_values=[1])
    del dataset
    del target_ds
    shp.Release()


if __name__ == "__main__":

    file_path = sys.argv[1]
    mask_path = sys.argv[2]
    result_output_path = sys.argv[3]
    print('矢量数据目录为：',str(sys.argv[1]))
    print('掩膜数据目录为：',str(sys.argv[2]))
    print('栅格输出目录为：',str(sys.argv[3]))
    shp_raster(file_path,mask_path,result_output_path)
    print('The successful running')
