# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:39:28 2020

@author: Yunzhao Ge
"""


# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.pyplot import MultipleLocator

import sys
import gdal
#from numpy import ma


def get_extrapolation_grid(img,mask,max_iter = 20,weight_window = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype = float),backgroud_value = None):
    img = img.copy()
    if backgroud_value == None:
        backgroud_value = img[mask].mean()
    img[~mask] = backgroud_value
    window_half_height = int((weight_window.shape[0] - 1)/2)
    window_half_width = int((weight_window.shape[1] - 1)/2)
    weight_sum = weight_window.sum()
    ext_shape = (img.shape[0] + weight_window.shape[0] - 1, img.shape[1] + weight_window.shape[1] - 1)
    ext_img = get_ext_shape_grid(img,ext_shape,fill_value = backgroud_value)
    row_ids,col_ids = np.where(mask < 1)
    raw_in_ext_origin_row_id = int((ext_shape[0] - img.shape[0])/2)
    raw_in_ext_origin_col_id = int((ext_shape[1] - img.shape[1])/2)
    while max_iter > 0:
        new_ext_img = ext_img.copy()
        for i in range(len(row_ids)):
            col_id = col_ids[i]
            row_id = row_ids[i]
            ext_position_left = col_id + raw_in_ext_origin_col_id - window_half_width
            ext_position_right = col_id + raw_in_ext_origin_col_id + window_half_width +1
            ext_position_upper = row_id + raw_in_ext_origin_row_id - window_half_height
            ext_position_down = row_id + raw_in_ext_origin_row_id + window_half_height + 1
            values = ext_img[ext_position_upper:ext_position_down,ext_position_left:ext_position_right]
            new_value = (values*weight_window).sum()/weight_sum
            new_ext_img[row_id + raw_in_ext_origin_row_id,col_id + raw_in_ext_origin_col_id] = new_value
        ext_img = new_ext_img.copy()
        max_iter = max_iter - 1
    return ext_img[window_half_height:-window_half_height,window_half_width:-window_half_width]

def get_ext_shape_grid(img,new_shape,fill_value = 0):
    new_2d_array = np.zeros(new_shape,dtype = img.dtype)   #（m+14,n+14） 0
    new_2d_array.fill(fill_value)                             # 填充为 0
    img_shape = img.shape                                     # (230, 185)
    row_extend_upper_lenth = int((new_shape[0] - img_shape[0])/2)  #  7
    col_extend_left_lenth = int((new_shape[1] - img_shape[1])/2)   #  7
    new_2d_array[row_extend_upper_lenth:row_extend_upper_lenth + img_shape[0],col_extend_left_lenth : col_extend_left_lenth + img_shape[1]] = img
    return new_2d_array   # 上下左右套了 7 的img

def get_ext_shape_mask(img,new_shape):
    new_2d_array = np.zeros(new_shape,dtype = bool)     #（m+14,n+14） 0
    img_shape = img.shape                                  # (m,n)
    row_extend_upper_lenth = int((new_shape[0] - img_shape[0])/2)   # 7
    col_extend_left_lenth = int((new_shape[1] - img_shape[1])/2)    # 7
    new_2d_array[row_extend_upper_lenth:row_extend_upper_lenth + img_shape[0],col_extend_left_lenth : col_extend_left_lenth + img_shape[1]] = img
    return new_2d_array   # 上下左右套了 7 的 img_mask

def get_raw_shape_grid(img,raw_shape):
    extend_shape = img.shape
    row_extend_upper_lenth = int((extend_shape[0] - raw_shape[0])/2)
    col_extend_left_lenth = int((extend_shape[1] - raw_shape[1])/2)
    raw_img = img[row_extend_upper_lenth:row_extend_upper_lenth + raw_shape[0],col_extend_left_lenth : col_extend_left_lenth + raw_shape[1]]
    return raw_img

def get_values_in_ext_grid(position_in_raw,ext_grid,raw_shape,window_shape):
    ext_shape = ext_grid.shape
    window_half_height = int((window_shape[0]-1)/2)
    window_half_width = int((window_shape[1]-1)/2)
    raw_in_ext_origin_row_id = int((ext_shape[0] - raw_shape[0])/2)
    raw_in_ext_origin_col_id = int((ext_shape[1] - raw_shape[1])/2)
    ext_position_left = position_in_raw[1] + raw_in_ext_origin_col_id - window_half_width
    ext_position_right = position_in_raw[1] + raw_in_ext_origin_col_id + window_half_width +1
    ext_position_upper = position_in_raw[0] + raw_in_ext_origin_row_id - window_half_height
    ext_position_down = position_in_raw[0] + raw_in_ext_origin_row_id + window_half_height + 1
    #print ext_position_left,ext_position_right,ext_position_upper,ext_position_down
    return ext_grid[ext_position_upper:ext_position_down,ext_position_left:ext_position_right]

def get_exist_values(position,img,window_shape):
    raw_shape = img.shape
    ext_shape = (img.shape[0] + window_shape[0],img.shape[1] + window_shape[1])
    ext_grid = get_ext_shape_grid(img,ext_shape)
    values = get_values_in_ext_grid(position,ext_grid,raw_shape,window_shape)
    return values

def get_position_in_ext_grid(position_in_raw,ext_shape,raw_shape):
    raw_in_ext_origin_row_id = int((ext_shape[0] - raw_shape[0])/2)
    raw_in_ext_origin_col_id = int((ext_shape[1] - raw_shape[1])/2)
    ext_col_position = position_in_raw[1] + raw_in_ext_origin_col_id
    ext_row_position = position_in_raw[0] + raw_in_ext_origin_row_id
    #print ext_position_left,ext_position_right,ext_position_upper,ext_position_down
    return ext_row_position,ext_col_position

def get_classified_contour(in_bool_img):
    max_class_id = 1
    raw_shape = in_bool_img.shape
    ext_shape = (raw_shape[0] + 2,raw_shape[1] + 2)
    ext_contour = get_ext_shape_grid(in_bool_img,ext_shape,0)
    ext_classification = ext_contour.astype("int")
    for rowp_in_raw in range(raw_shape[0]):
        for colp_in_raw in range(raw_shape[1]):
            rowp_in_ext, colp_in_ext = get_position_in_ext_grid((rowp_in_raw,colp_in_raw),ext_shape,raw_shape)
            square = ext_classification[rowp_in_ext - 1:rowp_in_ext + 1,colp_in_ext - 1:colp_in_ext + 1]
            #print rowp_in_raw,colp_in_raw
            #print rowp_in_ext,colp_in_ext
            square_set = set(square.ravel())
            if square[1,1] == 1:
                try:
                    square_set.remove(0)
                except:
                    pass
                if len(square_set) == 3:
                    square_set_array = np.array(list(square_set),dtype = "int")
                    square_set_array.sort()
                    ext_classification[rowp_in_ext, colp_in_ext] = square_set_array[2]
                    ext_classification[ext_classification == square_set_array[1]] = square_set_array[2]
                elif len(square_set) == 2:
                    square_set_array = np.array(list(square_set),dtype = "int")
                    ext_classification[rowp_in_ext, colp_in_ext] = square_set_array.max()
                else:
                    max_class_id = max_class_id + 1
                    ext_classification[rowp_in_ext, colp_in_ext] = max_class_id
    classification = get_raw_shape_grid(ext_classification,raw_shape)
    class_ids = list(set(classification.ravel()))
    class_ids = np.array(class_ids)
    class_ids.sort()
    class_ids = class_ids[1:]
    for class_id in class_ids:
        class_id_mask = np.where(classification == class_id,1,0)
        class_id_mask = class_id_mask.astype(bool)
        class_id_count = class_id_mask.sum()
        classification[class_id_mask] = class_id_count
    return classification

def get_classifications_img_to_thresholds(img,trreshold_min,threshold_max):
    thresholds = np.logspace(np.log10(trreshold_min),np.log10(threshold_max),num = 20)
    classifications = dict()
    for threshold in thresholds:
        contour = np.where(img > threshold,1,0)
        classification = get_classified_contour(contour)
        classifications[threshold] = classification
    return classifications

class Property_Window(object):
    def __get__(self, obj, objtype):
        return obj._window

    def __set__(self, obj, window):
        obj._window = window
        obj._window_shape = obj._window.shape
        minium_ext_shape = (obj._window.shape[0] + obj.img.shape[0] - 1, obj._window.shape[1] + obj.img.shape[1] - 1)

        if minium_ext_shape[0] >= obj._ext_shape[0]:
            new_minium_ext_shape_row = minium_ext_shape[0]
        else:
            new_minium_ext_shape_row = obj._ext_shape[0]
        if minium_ext_shape[1] >= obj._ext_shape[1]:
            new_minium_ext_shape_col = minium_ext_shape[1]
        else:
            new_minium_ext_shape_col = obj._ext_shape[1]
        obj._ext_shape = (new_minium_ext_shape_row,new_minium_ext_shape_col)

class Property_Mask(object):
    def __get__(self, obj, objtype):
        return obj._img_mask

    def __set__(self, obj, mask):
        obj._img_mask = mask                                             #1  obj._img_mask = (m,n) 的 T F
        obj.ext_img_mask = get_ext_shape_mask(obj._img_mask,obj._ext_shape) #1 obj._ext_shape (m+14,n+14)
                                                         #  obj.ext_img_mask    上下左右套了 7 的 img_mask

class Property_Extshape(object):

    def __get__(self, obj, objtype):
        return obj._ext_shape

    def __set__(self, obj, ext_shape):
        ext_update = False
        if obj._ext_shape[0] >= ext_shape[0]:         #2 此时 成立
            new_ext_shape_row = obj.ext_shape[0]      #2 new_ext_shape_row 为m+14
        else:
            new_ext_shape_row = ext_shape[0]         #1   m + 14
            ext_update = True
        if obj._ext_shape[1] >= ext_shape[1]:         #2 此时 成立
            new_ext_shape_col = obj._ext_shape[1]     #2 new_ext_shape_col 为n+14
        else:
            new_ext_shape_col = ext_shape[1]         #1   n  +  14
            ext_update = True
        obj._ext_shape = (new_ext_shape_row,new_ext_shape_col)   #1、2  obj._ext_shape  =  m+14 , n+14

        if ext_update:

            try:
                obj.ext_img_mask = get_ext_shape_mask(obj._img_mask,obj._ext_shape)  #1目前 sigularity.grid_array无 _img_mask该属性，则 运行except的pass
            except:
                pass
            if obj._extrapol == False:                                               #1 目前是 False,则可运行
                obj.ext_img = get_ext_shape_grid(obj.img,obj._ext_shape,fill_value = obj.ext_fill_value)   #1 上下左右套了7的img
                #  obj.img nodata 为0 ； obj._ext_shape （m+14 , n+14）；obj.ext_fill_value = 0
            else:
                ext_img_temp = get_ext_shape_grid(obj.img,obj._ext_shape,fill_value = obj.ext_fill_value)
                ext_extrapol_img = get_extrapolation_grid(ext_img_temp,obj.ext_img_mask,max_iter = obj.extrapol_max_iter,weight_window =obj.extrapol_weight_window,backgroud_value = obj.extrapol_backgroud_value)
                obj.ext_img = ext_extrapol_img

class Property_Extrapol(object):

    def __get__(self, obj, objtype):
        return obj._extrapol
    def __set__(self,obj, val):
        if val == False:
            obj.ext_img = get_ext_shape_grid(obj.img,obj.ext_shape,fill_value = obj.ext_fill_value)
            obj._extrapol =False
        else:
            if obj._extrapol == False:
                ext_img_temp = get_ext_shape_grid(obj.img,obj.ext_shape,fill_value = obj.ext_fill_value)
                ext_extrapol_img = get_extrapolation_grid(ext_img_temp,obj.ext_img_mask,max_iter = obj.extrapol_max_iter,weight_window =obj.extrapol_weight_window,backgroud_value = obj.extrapol_backgroud_value)
                obj.ext_img = ext_extrapol_img
            else:
                pass
            obj._extrapol = True

class Grid(object):
    window = Property_Window()
    img_mask = Property_Mask()
    ext_shape = Property_Extshape()
    extrapol = Property_Extrapol()

    def __init__(self,img,img_mask = None,ext_shape = None, window = np.array([[1]],dtype = bool),ext_fill_value = None):
        self.extrapol_max_iter = 20                               # self → singularity.grid_array
        self.extrapol_weight_window = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype = float)   # 3*3 的 1
        self.extrapol_backgroud_value = None
        self._ext_shape = (1,1)
        self._extrapol = False
        img = img.copy()
        if ext_fill_value == None:
            if img_mask == None:
                self.ext_fill_value = img.mean()
            else:
                self.ext_fill_value = img[img_mask].mean()
                img[~img_mask] = self.ext_fill_value
        else:
            self.ext_fill_value = ext_fill_value                   # 拓展填充部分为0
            img[~img_mask] = self.ext_fill_value                   # 将img 的 nodata 填充为0

        self.img = img

        minium_ext_shape = (window.shape[0] + self.img.shape[0] - 1, window.shape[1] + self.img.shape[1] - 1)# 行列均增加14
        self.ext_shape = minium_ext_shape                    # 转到 → Property_Extshape(object):
        # 通过Property_Extshape 的__set__, obj._ext_shape 变成（m+14,n+14）  obj.ext_img 变成 上下左右套了 7的img

        if ext_shape is None:                   # 此时是 None
            self.ext_shape = minium_ext_shape   # 转到 → Property_Extshape(object):
            #得到 obj._ext_shape 为 m+14,n+14

        else:
            self.ext_shape = ext_shape
        if img_mask is None:
            self.img_mask = np.ones_like(img,dtype = bool)
        else:                                   # img_mask 不是 None:   调用 Property_Mask
            self.img_mask = img_mask            # self.img_mask   也就是return self._img_mask

        self.window = window                    # self.window      15*15的 T

        self.ext_img_mask = get_ext_shape_mask(self.img_mask,self.ext_shape) # self.ext_img_mask 为上下左右均套了7的img_mask

    def __call__(self,row_id,col_id):
        window_half_height = int((self.window.shape[0]-1)/2)    #   7
        window_half_width = int((self.window.shape[1]-1)/2)     #   7
        raw_in_ext_origin_row_id = int((self.ext_shape[0] - self.img.shape[0])/2)    # 7
        raw_in_ext_origin_col_id = int((self.ext_shape[1] - self.img.shape[1])/2)    # 7
        ext_position_left = col_id + raw_in_ext_origin_col_id - window_half_width
        ext_position_right = col_id + raw_in_ext_origin_col_id + window_half_width +1
        ext_position_upper = row_id + raw_in_ext_origin_row_id - window_half_height
        ext_position_down = row_id + raw_in_ext_origin_row_id + window_half_height + 1
                                                        # 以 row_id  col_id 为中心的窗口的四个边界
        #print ext_position_left,ext_position_right,ext_position_upper,ext_position_down
        return self.ext_img[ext_position_upper:ext_position_down,ext_position_left:ext_position_right]  # 返回其要处理的那一块

class Window():
    def __init__(self,full_size_half):

        self.full_size = full_size_half*2 +1                    # 15
        self.full_size_half = full_size_half                    # 7
        self.window = np.zeros((self.full_size,self.full_size),dtype = bool)  # 15*15 True
        self.win_max = self.__call__(self.full_size_half)                        # 15*15 True


    def __call__(self,small_size_half):#square window
        down_limit = self.full_size_half - small_size_half
        up_limit = self.full_size_half + small_size_half            
        for row in range(self.full_size):
            for col in range(self.full_size):
                self.window[row,col] = (row >= down_limit)*((row <= up_limit))*(col >= down_limit)*((col <= up_limit))
        return self.window

class Singularity():
    def __init__(self,array2d,mask2d,win,win_size_list):
        self.minium_win_size_half = 3
        self.win_size_list = win_size_list
        self.win_full_size_half = win.full_size_half               # 7
        self.win_size_list.sort()                                  # 排序
        self.nof_win = len(self.win_size_list)                     # 8
        self.win_max = win.win_max                                 # 15*15 True
        self.grid_array = Grid(array2d,mask2d,window = self.win_max,ext_fill_value = 0)
        self.grid_mask = Grid(mask2d,mask2d,window = self.win_max,ext_fill_value = 0)
        self.win_core = win(self.win_size_list[0]).copy()
        self.win_core_area = self.win_core.sum()
        #print "self.win_core_area:" + str(self.win_core_area)
        #print self.win_core
        #print self.win_core.sum()
        self.win_list = [self.win_core.copy()]
        self.win_incre_list = [self.win_core.copy()]
        self.win_area_list = [self.win_core_area]
        self.win_incre_area_list = [self.win_core_area]
        for i in range(len(self.win_size_list) - 1):
            win_size = self.win_size_list[i + 1]
            #print "win_size"
            #print win_size
            win_temp = win(win_size)
            self.win_list.append(win_temp.copy())
            self.win_area_list.append(win_temp.sum())
            win_incre = win_temp ^ self.win_list[i]
            self.win_incre_list.append(win_incre.copy())
            self.win_incre_area_list.append(win_incre.sum())
        
        
        
    def cal(self):
        nof_p = len(self.win_density_list)
        density_array = np.array(self.win_density_list,dtype = float)
        area_array = np.array(self.win_area_list[:nof_p])
        density_array_log = np.log2(density_array)
        #print(area_array)
        area_array_log = np.log2(area_array)
        #print(area_array_log.shape)
        #print(density_array_log.shape)
        area_array_log2 = np.vstack([area_array_log, np.ones(nof_p)]).T


        a, c = np.linalg.lstsq(area_array_log2, density_array_log,rcond = None)[0]   # a 斜率  c 截距

        return 0 - a, c
        
        
        
    def __call__(self,row,col):
        win_data_max = self.grid_array(row,col)    # 返回其要处理的那一块数据  15*15
        win_mask_max = self.grid_mask(row,col)     # 返回其要处理的那一块数据对应的掩膜  15*15
        cell_value = win_data_max[self.win_full_size_half,self.win_full_size_half]   # 核心
        core = win_mask_max*self.win_list[0]       #      中间为1 其余为0 的15*15
        core_area = core.sum()                     #      core_area 为 1
        if self.win_core_area -core_area > 0:      #      否
            #print self.win_core_area -core_area
            #print "miss core"
            alpha = 0
            return alpha,cell_value
        else:
            #win_area_list = [self.win_core_area]
            #win_mask_area_list = [core_area]
            core_mass = (win_data_max*core).sum()          #  中间那个数
            core_density = core_mass/self.win_core_area    #  value_core/1 = value_core
            self.win_density_list = [core_density]
            self.win_mass_list = [core_mass]
        for i in range(self.nof_win - 1):
            j = i + 1                                      #1  j = 1   2
            win_incre_mask = self.win_incre_list[j]*win_mask_max   #1  看哪些地方是有值的 bool
            win_incre_mask_area = win_incre_mask.sum()             #1  看这一圈有值的个数
            if win_incre_mask_area > 0:                            #True
                win_incre_mass = win_data_max[win_incre_mask].sum() #计算第一圈值的总和
                win_incre_ext_mass = win_incre_mass*self.win_incre_area_list[j]/win_incre_mask_area# 若都有值，理应为多少
                #print "i:" + str(i)
                #print "win_mass_list:" + str(self.win_mass_list)
                win_mass = win_incre_ext_mass + self.win_mass_list[i]   # 现在的总和（9个数）
                win_density = win_mass/self.win_area_list[j]            # 平均的个数
                self.win_density_list.append(win_density)               #
                self.win_mass_list.append(win_mass)                     #
            else: #win_incre_mask_area <= 0
                if j <= 1:
                    alpha = 0
                    #print "data is too less"
                    return alpha, cell_value
                else:
                    #print "window is too large"
                    alpha, fractal_density = self.cal()
                    return alpha, fractal_density
        alpha, fractal_density = self.cal()
       # print "normal"
        return alpha, fractal_density



def cal_singularity(img,img_mask,window_size_max,window_size_squence):
    singularity_array = img_mask.astype("float")               # 掩膜的 0  1
    fractal_density_array = img_mask.astype("float")           # 掩膜的 0  1
    window = Window(window_size_max)                           # window 为一 class
    singularity = Singularity(img,img_mask,window,window_size_squence)   #
    exist_row_ids,exist_col_ids = np.where(img_mask == True)          #  有值的横纵坐标
    nof_cells = len(exist_row_ids)                                       #  非nodata值的个数
    for i in range(nof_cells):
        #print str(exist_row_ids[i])+ ":"+str(exist_col_ids[i])
        a,c = singularity(exist_row_ids[i],exist_col_ids[i])             #  代入singularity类的 __call__ 函数
        #print "a:" + str(a) +"|c:" + str(c)
        singularity_array[exist_row_ids[i],exist_col_ids[i]] = a
        fractal_density_array[exist_row_ids[i],exist_col_ids[i]] = c
    return singularity_array,fractal_density_array




def matplotlib(x,y,img_reversed,path_plot):
    plt.figure()
    cmap = plt.get_cmap('jet')
#cs = plt.contourf(x,y,img_reversed,cmap = cmap)
    cs = plt.pcolormesh(x,y,img_reversed,cmap = cmap)
    cbar = plt.colorbar(cs)
#plt.title('PCA1',fontsize = 28) 
    font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        'rotation':'vertical'}
    cbar.set_label('Sigularity_value',fontdict=font)
#plt.text(round(dimension[2]/2),-8,'PCA1',fontsize = 38)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Local singularity analysis' ,fontsize = 38)
#plt.show()
#plt.savefig(r'C:\Users\Administrator\Desktop\PCAA.png',bbox_inches='tight',dpi = 72)
    plt.savefig(path_plot)
    
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

#===============================================================


if __name__ == '__main__':
    file_path = sys.argv[1]
    mask_path = sys.argv[2]
    result_output_path_npy = sys.argv[3]
    result_output_path_plot = sys.argv[4]
    result_output_path_tif = sys.argv[5]

    mask = gdal.Open(mask_path)
    img_mask = mask.GetRasterBand(1).ReadAsArray()
    img_mask = np.where(img_mask==0,False,True)

    value = gdal.Open(file_path)
    img   = value.GetRasterBand(1).ReadAsArray()

    window_size_max = 7
    window_size_squence = [0,1,2,3,4,5,6,7]

    rows,columns = img_mask.shape
    dx, dy = 1, 1
    y, x = np.mgrid[slice(1, rows + dy, dy),slice(1,columns + dx, dx)]

    plt.rcParams['figure.figsize'] = (columns/rows*24,24)
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24

    singularity_array,fractal_density_array = cal_singularity(img,img_mask,window_size_max,window_size_squence)
    
    np.save(result_output_path_npy,singularity_array)

    write_geotiff(result_output_path_tif, singularity_array, mask.GetGeoTransform(),
                  mask.GetProjectionRef())

    singularity_array[~img_mask] = np.nan
    img = pd.DataFrame(singularity_array)
    
    img_reversed = img.iloc[list(reversed(range(rows)))]
    matplotlib(x,y,img_reversed,result_output_path_plot)
    print('The successful running')









