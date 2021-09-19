# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 23:40:24 2020

@author: LGSYRZ
"""

import sys,geopandas as gpd
if __name__ == '__main__':
    file_path = sys.argv[1]
    field = sys.argv[2]
    value = sys.argv[3]
    result_output_path_shp = sys.argv[4]
    print('矢量数据目录为：',str(sys.argv[1]))
    print('所选字段为：',str(sys.argv[2]))
    print('唯一值为：',str(sys.argv[3]))
    print('矢量数据输出路径为：',str(sys.argv[4]))    
    Table = gpd.read_file(file_path)
    selection = Table.loc[Table[field]==value]
    selection.to_file(result_output_path_shp)
    #print(selection)
    #file_path=r'F:\mineral prospecting\Attribute\Input\Cu.shp'
    #result_output_path_shp = r'F:\mineral prospecting\Attribute\Output\part2\Cu.shp'
    print('The successful running')
    