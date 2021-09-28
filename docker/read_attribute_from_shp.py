# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:47:26 2020

@author: Yunzhao Ge
"""

import geopandas as gpd
import sys

if __name__ == '__main__':
    file_path = sys.argv[1]
    result_output_path_csv = sys.argv[2]
    result_output_path_txt = sys.argv[3]
    print('矢量数据目录为：',str(sys.argv[1]))
    print('属性表输出路径：',str(sys.argv[2]))
    print('唯一值输出路径：',str(sys.argv[3]))
    Table = gpd.read_file(file_path)
    del Table['geometry']
    Table.to_csv(result_output_path_csv)
    f = open(result_output_path_txt,'a')
    f.write('The unique values for each field in the Table are shown below:\n')
    for i,j in enumerate(Table.columns):
        f.write('----------'+'Field'+str(i)+'-----'+j+'---------------\n')
        f.write(str(Table[j].unique())+'\n')
        f.write('  \n')
    f.close()
    print('The successful running')







