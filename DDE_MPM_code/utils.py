import numpy as np,gdal
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


def remove_backgrounds(data, label):
    label = label.reshape(-1)
    data = data.reshape(-1, data.shape[-1])

    fg_idx = label != -1
    data_new = data[fg_idx, :]
    label_new = label[fg_idx]

    return data_new, label_new


def get_training_functions(Method, label, w_p2n, *args, **kwargs):
    if Method == 'AdaBoost':
        clf = AdaBoostClassifier(*args, **kwargs)
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.predict_proba
        weight = label*(w_p2n-1)+1
    elif Method == 'DecisionTree':
        clf = DecisionTreeClassifier(*args, **kwargs)
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.predict_proba
        weight = label*(w_p2n-1)+1
    elif Method == 'LogisticRegression':
        clf = LogisticRegression(*args, **kwargs)
        trainFunc = clf.fit
        testFunc = clf.score
        # outputFunc = clf.decision_function
        outputFunc = clf.predict_proba
        weight = label*(w_p2n-1)+1
    elif Method == 'NaiveBayes':
        clf = GaussianNB(*args, **kwargs)
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.predict_proba
        weight = label*(w_p2n-1)+1
    elif Method == 'RandomForest':
        clf = RandomForestClassifier(*args, **kwargs)
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.predict_proba
        weight = label*(w_p2n-1)+1
    elif Method == 'SVM':
        clf = SVC(*args, **kwargs)
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.decision_function
        # outputFunc = clf.predict_proba
        weight = label*(w_p2n-1)+1
        # weight = np.ones_like(label)
    else:
        print("Error Method" + Method + "!")
        return None, None, None
    return trainFunc, testFunc, outputFunc, weight


def compute_confusion_matrix(label, scores):
    y_pred = np.zeros_like(scores)
    if scores.min() >= 0:
        y_pred[scores > 0.5] = 1
    else:
        y_pred[scores > 0] = 1
    return confusion_matrix(label, y_pred)


def save_PR_curve(label, scores, Method, save_pr_file_name, save_f1_file_name, vis=False):
    precision, recall, thresholds = precision_recall_curve(label, scores)
    f1_score = 2 * ((precision + 1e-5) * (recall + 1e-5)) / (precision + recall + 2e-5)

    plt.figure()
    plt.plot(recall, precision, lw=2, label=Method)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc="lower right")
    if vis:
        plt.show()
    plt.savefig(save_pr_file_name)
    plt.close()
    #print('PR curve saved in ' + save_pr_file_name)

    plt.figure()
    plt.plot(np.arange(0, f1_score.shape[0])/f1_score.shape[0], f1_score, lw=2, label=Method)
    plt.xlabel('Thresh')
    plt.ylabel('F1-score')
    plt.title('F1-score curve')
    plt.legend(loc="lower right")
    if vis:
        plt.show()
    plt.savefig(save_f1_file_name)
    plt.close()
    #print('F1 curve saved in ' + save_f1_file_name)


def save_ROC_curve(label, scores, Method, save_file_name, vis=False):
    fpr, tpr, _ = roc_curve(label, scores)
    roc_auc = auc(fpr, tpr)
    print('AUC = ', roc_auc)
    plt.plot(fpr, tpr, lw=2, label=Method)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    if vis:
        plt.show()
    plt.savefig(save_file_name)
    plt.close()
    #print('ROC curve saved in ', save_file_name)
    return roc_auc


def write_results(filename, results, method, w_p2n, *args, **kwargs):

    import csv
    tmp = open(filename, 'a', newline='', encoding="utf-8")  # a表示在最后一行后面追加

    csv_write = csv.writer(tmp)
    csv_write.writerow([' ', ' ', ' ', ' ', method + ' Report'])
    csv_write.writerow(' ')

    # write parameters
    csv_write.writerow(['Parameters'])
    csv_write.writerow([f'w_p2n={w_p2n}', kwargs or 'Defalt Parameters'])
    csv_write.writerow(' ')
    csv_write.writerow(' ')

    # write ACC
    csv_write.writerow(['ACC'])
    csv_write.writerow([results['ACC']])
    csv_write.writerow(' ')
    csv_write.writerow(' ')

    # write AUC
    csv_write.writerow(['AUC'])
    csv_write.writerow([results['AUC']])
    csv_write.writerow(' ')
    csv_write.writerow(' ')

    # write Confusion Matrix
    csv_write.writerow(['Confusion Matrix'])
    for row in range(results['Confusion Matrix'].shape[0]):
        item = results['Confusion Matrix'][row]
        csv_write.writerow(item)
    csv_write.writerow(' ')
    csv_write.writerow(' ')

    # write score
    csv_write.writerow(['score'])
    for row in range(results['score'].shape[0]):
        item = results['score'][row]
        csv_write.writerow(item)
    csv_write.writerow(' ')
    csv_write.writerow(' ')

    tmp.close()


def photo(x,y,iimg_reversed,output_dir_png,name):
    plt.figure()
    cmap = plt.get_cmap('jet')
    cs = plt.pcolormesh(x, y, iimg_reversed, cmap=cmap)
    cbar = plt.colorbar(cs)
    #plt.title('S-A-%s' % name, fontsize=28)
    font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 24, 'rotation': 'vertical'}
    cbar.set_label('%s' % name, fontdict=font)
    # plt.text(round(dimension[2]/2),-8,'PCA1',fontsize = 38)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('%s' % name, fontsize=38)
    #plt.show()
    plt.savefig(output_dir_png)
def column_normalization(data):
    data_mean = data.mean(0).reshape(1, -1)
    data_var = data.var(0).reshape(1, -1)
    data = (data - data_mean) / data_var ** 0.5
    return data

def write_geotiff(fname, data, geo_transform, projection,Nodata = -9999):

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, eType = gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(Nodata)
    dataset = None



def data_norm(data):
    return column_normalization(data)