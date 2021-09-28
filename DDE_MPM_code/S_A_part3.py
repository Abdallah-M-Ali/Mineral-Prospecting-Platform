# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:21:04 2020

@author: Yunzhao Ge
"""
import sys,numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn import  linear_model


def line(x,y,x_train):
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    n = regr.coef_
    n = n[0]
    n = np.around(n, decimals=3)
    m = regr.intercept_
    m = np.around(m, decimals=3)
    r = regr.score(x, y)
    r = np.around(r, decimals=3)
    y_train = regr.predict(x_train)
    return n[0],m[0],r,y_train

def photo1(num,points_part,text=[]):
    plt.figure()
    for i in range(len(num) - 1):
        text_seg = [i+1]
        downs_select = points_part[points_part[:, 0] > num[i]]
        downs_select = downs_select[downs_select[:, 0] < num[i + 1]]
        text_seg.append(round(downs_select[-1,0],3))
        text_seg.append(round(np.e**downs_select[-1,0],3))
        x = downs_select[:, 0].reshape(-1, 1)
        y = downs_select[:, 1].reshape(-1, 1)
        x_train = np.arange(num[i], num[i + 1], 0.001)
        x_train = x_train.reshape([x_train.shape[0], 1])
        n, m, r,y_train = line(x, y, x_train)
        text_seg.append(n)
        text_seg.append(m)
        text_seg.append(r)
        if i>0:
            plt.scatter(x, y, color='green', linewidths=0.001)
            plt.plot(x_train, y_train, color='red', linewidth=2)
        plt.ylabel('Log(Area)')
        plt.xlabel('Log(Value)')
        text.append(text_seg)
        plt.plot([downs_select[-1, 0], downs_select[-1, 0]], [downs_select[-1, 1] - 1.5, downs_select[-1, 1] + 0.5], color='blue',
                     linewidth=2)
    plt.grid(linestyle='-.',which='major')
    labels = ['Seg','Log(Value)','value','K','b',r'$ R^2 $']
    text.insert(0,labels)
    #my_table = plt.table(cellText = text,cellLoc = 'center',colWidths = [0.12]*6,
    #                  bbox = [0.03, 0.03, 0.82, 0.48])
    #my_table.auto_set_font_size(False)
    #my_table.set_fontsize(10)
    plt.savefig(result_output_plot)
    return text







# file_path = r'F:\mineral prospecting\S-A\Output\Part2\Npy\s-a.npy'
# point = '-10,5,11,14'
# result_output_plot = r'F:\mineral prospecting\S-A\Output\Part3\Plot\s-a.png'
# result_output_csv = r'F:\mineral prospecting\S-A\Output\Part3\Data\s-a.csv'


if __name__ == '__main__':
    file_path = sys.argv[1]
    point = sys.argv[2]
    result_output_plot = sys.argv[3]
    result_output_csv = sys.argv[4]
    num = [float(x) for x in point.split(',')]
    points_part = np.load(file_path)
    text = photo1(num, points_part)
    text_np = np.array(text)
    pd.DataFrame(text_np).to_csv(result_output_csv)
    print('The successful running')
    




