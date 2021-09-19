# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:18:43 2020

@author: LGSYRZ
"""


import ogr, os,sys

def createBuffer(inputfn, outputBufferfn, bufferDist):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()             # 获得句柄
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None

def main(inputfn, outputBufferfn, bufferDist):
    createBuffer(inputfn, outputBufferfn, bufferDist)

if __name__ == "__main__":
    file_path = sys.argv[1]
    distance = sys.argv[2]
    output_path = sys.argv[3]
    
    print('矢量数据目录为：',str(sys.argv[1]))
    print('缓冲距离为：',sys.argv[2]+'m')
    print('矢量数据输出路径：',str(sys.argv[3]))
    
    main(file_path, output_path, int(distance))
    print('The successful running')


