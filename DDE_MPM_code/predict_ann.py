import pandas as pd
import numpy as np,gdal
import matplotlib.pyplot as plt

from utils import remove_backgrounds
from utils import get_training_functions
from utils import save_ROC_curve, save_PR_curve, compute_confusion_matrix
from ANN.main_ANN import trainANN
import os
from utils import write_results
from utils import data_norm
from utils import photo,write_geotiff
import sys

def predict_ann(data_all, output_dir, result_dir, batch_size=128, learning_rate=0.001):

    mask_raw = data_all[:, :, 0].copy()
    label_raw = data_all[:, :, 1].copy()
    data_raw = data_all[:, :, 2:].copy()
    
    data_raw[mask_raw == 0, :] = 0

    nDim = data_raw.shape[2]
    w = data_raw.shape[0]
    h = data_raw.shape[1]

    mask_unique = np.unique(mask_raw)
    label_unique = np.unique(label_raw)
    if mask_unique.size != 2  or mask_unique[0] != 0  or mask_unique[1] != 1 or\
       label_unique.size != 2 or label_unique[0] != 0 or label_unique[1] != 1:
        return -1

    label_raw[mask_raw == 0] = -1

    # remove useless data
    data, label = remove_backgrounds(data_raw, label_raw)

    results = {
        'score': None,
        'Confusion Matrix': None,
        'ACC': None,
        'AUC': None,
        'ROC': None,
        'PR': None,
        'F1-score': None
    }

    # Neural Networks
    NN_results = trainANN(data_raw, label_raw, output_dir, batch_size=batch_size, learning_rate=learning_rate)

    # NN outputs
    results['score'] = NN_results['score']
    results['Confusion Matrix'] = NN_results['Confusion Matrix']
    results['ACC'] = NN_results['ACC']
    results['AUC'] = NN_results['AUC']
    results['ROC'] = NN_results['ROC']
    results['PR'] = NN_results['PR']
    results['F1-score'] = NN_results['F1-score']

    result_output_file_name = result_dir+'/report_ANN.csv'
    results['Report Path'] = result_output_file_name
    write_results(result_output_file_name, results, 'ANN', 'default', batch_size=batch_size, learning_rate=learning_rate)

    return results

if __name__ == '__main__':
    parameter = sys.argv[1]
    mask_path = sys.argv[2]
    label_path = sys.argv[3]
    file_list = sys.argv[4]
    output_dir= sys.argv[5]
    result_dir = sys.argv[6]
    output_dir_png = sys.argv[7]
    output_dir_tif = sys.argv[8]

    batch_size = 256
    learning_rate = 0.001

    parameter = parameter.split(',')
    for i in parameter:
        exec(i)
    file_list = file_list.split(',')

    file_value = []
    mask_tif = gdal.Open(mask_path)
    mask = mask_tif.GetRasterBand(1).ReadAsArray()
    file_value.append(mask)
    label = np.load(label_path)
    file_value.append(label)

    for j in file_list:
        value = np.load(j)
        value[np.isnan(value)] = 0
        file_value.append(value)
    sample = np.array(file_value)
    dimension = sample.shape
    sample = sample.transpose((1,2,0))
    out = predict_ann(sample, output_dir,result_dir,batch_size = batch_size,learning_rate = learning_rate)

    score = out['score']
    dimension = score.shape
    write_geotiff(output_dir_tif, score, mask_tif.GetGeoTransform(),
                  mask_tif.GetProjectionRef())

    dx, dy = 1, 1
    y, x = np.mgrid[slice(1, dimension[0] + dy, dy), slice(1, dimension[1] + dx, dx)]
    plt.rcParams['figure.figsize'] = (dimension[1] / dimension[0] * 24, 24)
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24

    score = np.where(score==-9999,np.nan,score)
    score = pd.DataFrame(score)
    score = score.iloc[list(reversed(range(dimension[0])))]
    photo(x,y,score,output_dir_png,'ANN')

