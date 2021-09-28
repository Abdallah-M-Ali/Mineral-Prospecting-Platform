# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:18:22 2020

@author: Yunzhao Ge
"""
import sys,numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def photo1(points_part):
    #plt.figure(figsize = (24,16))
    plt.figure()
    plt.scatter(points_part[:, 0], points_part[:, 1], color='green', linewidths=0.21)
    plt.ylabel('Log(Area)')
    plt.xlabel('Log(Value)')
    x_major_locator = MultipleLocator(0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
        #my_table.scale(1,4)
    plt.grid(linestyle='-.',which='major')
    plt.savefig(result_output_plot)



if __name__ == '__main__':
    file_path = sys.argv[1]
    left_endpoint = float(sys.argv[2])
    right_endpoint = float(sys.argv[3])
    result_output_plot = sys.argv[4]
    result_output_npy = sys.argv[5]
    points_part = np.load(file_path)
    downs = points_part[points_part[:, 0] >= float(left_endpoint)]
    downs = downs[downs[:, 0] <= float(right_endpoint)]
    np.save(result_output_npy,downs)
    photo1(downs)
    print('The successful running')
    
    
    
    
       
