import pandas as pd
import numpy as np,gdal
import matplotlib.pyplot as plt

from utils import remove_backgrounds
from utils import get_training_functions
from utils import save_ROC_curve, save_PR_curve, compute_confusion_matrix

import os
from utils import write_results
from utils import data_norm
from utils import photo,write_geotiff
import sys

def predict_logisticregression(data_all, output_dir, result_dir, w_p2n=100, *args, **kwargs):

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

    # settings
    sklearn_Methods = ['LogisticRegression']

    # remove useless data
    data, label = remove_backgrounds(data_raw, label_raw)
    data = data_norm(data)

    results = {
        'score': None,
        'Confusion Matrix': None,
        'ACC': None,
        'AUC': None,
        'ROC': None,
        'PR': None,
        'F1-score': None
    }
    for method in sklearn_Methods:
        # path
        ROC_output_file_name = output_dir + '/ROC_' + method + '.png'
        PR_output_file_name = output_dir + '/PR_' + method + '.png'
        F1_output_file_name = output_dir + '/F1_' + method + '.png'
        result_output_file_name = result_dir + '/report_' + method + '.csv'
        trainFunc, testFunc, outputFunc, weight = get_training_functions(method, label, w_p2n, *args, **kwargs)

        if method == 'SVM':
            label = label * 2 - 1

        # train test and save scores
        print("Start Traning " +  method + "...")
        trainFunc(data, label, weight)
        ACC = testFunc(data, label)

        # save scores
        output = outputFunc(data_raw.reshape(-1, nDim))
        if output.ndim == 2:
            scores = output[:, 1].reshape(w, h)
        else:
            scores = output.reshape(w, h)
        if method == 'SVM':
            scores = 1 / (1 + np.exp(-scores))
        scores[mask_raw==0] = -9999
        results['score']  = scores

        # save curves
        if method == 'SVM':
            label = (label + 1) / 2
        output = outputFunc(data)
        if output.ndim == 2:
            scores_curves = output[:, 1]
        else:
            scores_curves = output
        AUC = save_ROC_curve(label, scores_curves, method, ROC_output_file_name)
        save_PR_curve(label, scores_curves, method, PR_output_file_name, F1_output_file_name)
        confusion_mat = compute_confusion_matrix(label, scores_curves)
        results['Confusion Matrix']  = confusion_mat
        results['ACC']  = ACC
        results['AUC']  = AUC
        results['ROC']  = ROC_output_file_name
        results['PR']  = PR_output_file_name
        results['F1-score']  = F1_output_file_name
        results['Report Path'] = result_output_file_name

    write_results(result_output_file_name, results, sklearn_Methods[0], w_p2n, *args, **kwargs)

    return results

if __name__ == '__main__':

    parameter = sys.argv[1]
    mask_path = sys.argv[2]
    label_path = sys.argv[3]
    file_list = sys.argv[4]
    output_dir = sys.argv[5]
    result_dir = sys.argv[6]
    output_dir_png = sys.argv[7]
    output_dir_tif = sys.argv[8]

    w_p2n = 200
    C = 100
    solver='lbfgs'

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
    out = predict_logisticregression(sample, output_dir,result_dir,w_p2n=w_p2n,C = C,solver = solver)

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
    photo(x,y,score,output_dir_png,'Logistic regression')













