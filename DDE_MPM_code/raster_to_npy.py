# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 01:07:52 2020

@author: Yunzhao Ge
"""

import sys,numpy as np
import gdal

if __name__ == '__main__':
    raster_path = sys.argv[1]
    raster_output_path_npy = sys.argv[2]
    raster = gdal.Open(raster_path)
    raster = raster.GetRasterBand(1).ReadAsArray()
    np.save(raster_output_path_npy,raster)
    print('The successful running')
