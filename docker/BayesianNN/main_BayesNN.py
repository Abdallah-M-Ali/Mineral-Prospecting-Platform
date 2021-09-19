from __future__ import print_function

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from BayesianNN.utils.BayesianModels.BayesianWsNet import BBBWsNet
import os
from utils import save_ROC_curve, save_PR_curve, compute_confusion_matrix
from utils import data_norm


def get_data_dim(dataset_root):
    file_names = os.listdir(dataset_root)
    dim = 0
    for file_name in file_names:
        if file_name[-3:] == 'csv' and file_name[:5] != 'label':
            dim += 1
    return dim


def load_CSV_file(file_name):
    tmp = np.loadtxt(file_name, dtype=np.str, delimiter=",")
    data = tmp[1:, 1:].astype(np.float)
    return data


def load_dataset(dataset_root, nDim, w, h):
    file_names = os.listdir(dataset_root)

    # load data
    data_final = np.zeros((nDim, w, h))
    data = []
    for file_name in file_names:
        if file_name[-3:] == 'csv' and file_name[:5] != 'label':
            data.append(load_CSV_file(dataset_root + file_name))

    # load label
    label = load_CSV_file(dataset_root + "label.csv")

    for idx in range(nDim):
        data_final[idx] = data[idx]
    return data_final, label.reshape(1, w, h)


def remove_backgrounds(data, label):
    label = label.reshape(-1)
    data = data.reshape(-1, data.shape[-1])

    fg_idx = label != -1
    data_new = data[fg_idx, :]
    label_new = label[fg_idx]

    return data_new, label_new


def get_beta(batch_idx, m, beta_type):
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        # beta = min(epoch / (num_epochs // 4), 1)
        print('pass')
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta


def elbo(out, y, kl, beta):
    w1 = (y == 0).sum() / (y == 1).sum()
    # w1 = 1
    w = torch.FloatTensor([1, w1])
    loss = F.cross_entropy(out, y, w)
    #     print(out, y, loss)
    # print(loss.data)
    return loss + beta * kl


def train(net, optimizer, epoch, bs, data, label):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    numTrainData = len(data)
    numTrainBatch = int(numTrainData / bs)
    iter_all = 0
    for idx in range(numTrainBatch):
        #         inputs, targets = data[idx*bs:(idx+1)*bs,:], label[idx*bs:(idx+1)*bs,:]
        inputs, targets = data[idx * bs:(idx + 1) * bs, :], label[idx * bs:(idx + 1) * bs]
        inputs, targets = torch.from_numpy(inputs).float(), torch.from_numpy(targets).long()
        inputs, targets = Variable(inputs), Variable(targets)
        # inputs, targets = inputs.cuda(), targets.cuda()
        if targets.sum() == 0:
            continue
        else:
            iter_all += 1
        optimizer.zero_grad()
        outputs, kl = net.probforward(inputs)
        loss = elbo(outputs, targets, kl, get_beta(epoch, len(data), "Standard"))
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[BayesNN TRAIN] Acc: {100. * correct / total:.3f}, Iter: {iter_all:d}')
    return net


def test(net, optimizer, epoch, bs, data, label):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    accuracy_max = 0
    numTestData = len(data)
    numTestBatch = int(numTestData / bs)
    with torch.no_grad():
        for idx in range(numTestBatch):
            #             inputs, targets = data[idx*bs:(idx+1)*bs,:], label[idx*bs:(idx+1)*bs,:]
            inputs, targets = data[idx * bs:(idx + 1) * bs, :], label[idx * bs:(idx + 1) * bs]
            inputs, targets = torch.from_numpy(inputs).float(), torch.from_numpy(targets).long()
            inputs, targets = Variable(inputs), Variable(targets)
            # inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net.probforward(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100. * correct / total
        print(f'[BayesNN TEST] Acc: {accuracy:.3f}')


################################################################################################
################################################################################################
def trainBayesNN(data_raw, label_raw, output_dir, batch_size=128, learning_rate=0.001):

    Method = 'wsBayesnet'
    use_all_dimension = True
    ROC_output_file_name = output_dir+'/ROC_'+Method+'.svg'

    # get dimension
    w = label_raw.shape[0]
    h = label_raw.shape[1]

    outputs = 2
    nDim = data_raw.shape[-1]
    inputs = nDim
    resume = False
    n_epochs = 200
    lr = learning_rate
    weight_decay = 0.0005
    num_samples = 1
    resize = 64
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    # batch_size = batch_size
    # percentage of training set to use as validation
    test_size = 0.33
    epochs = [80, 40]

    # load data
    # data_raw, label_raw = load_dataset(dataset_root, nDim, w, h)

    # remove useless data
    data, label = remove_backgrounds(data_raw, label_raw)
    data = data_norm(data)
    print('postive data: ', (label == 1).sum())

    # specify the image classes
    classes = ['positive', 'negative']

    # Architecture
    if (Method == 'wsBayesnet'):
        net = BBBWsNet(outputs, inputs)
    else:
        print('Error : Network should be WsNet')

    count = 0
    from torch.optim import Adam

    bs = batch_size
    for epoch in epochs:
        optimizer = Adam(net.parameters(), lr=lr)
        for _ in range(epoch):
            ## shuffle training data
            indices = list(range(len(data)))
            np.random.shuffle(indices)
            data = data[indices, :]
            #         label = label[indices,:]
            label = label[indices]
            net = train(net, optimizer, count, bs, data, label)
            test(net, optimizer, count, bs, data, label)
            count += 1
        lr /= 10

    ## visualization
    input_data = torch.from_numpy(data_raw.reshape(-1, nDim)).float()
    input_data = Variable(input_data)

    net.eval()
    scores, _ = net.probforward(input_data)
    scores = F.softmax(scores, dim=1)
    vis_scores = scores[:, 1]
    vis_scores = vis_scores.reshape(w, h)
    vis_scores[label_raw == -1] = -9999

    # print(vis_scores.shape)
    # np.savetxt(output_score_name, vis_scores.data.cpu().numpy(), delimiter=',')

    # ROC curve
    input_data = torch.from_numpy(data.reshape(-1, nDim)).float()
    input_data = Variable(input_data)
    net.eval()
    scores_curves, _ = net.probforward(input_data)
    scores_curves = F.softmax(scores_curves, dim=1)[:, 1].detach().numpy()
    y_pred = np.zeros_like(scores_curves)
    y_pred[scores_curves > 0.5] = 1

    method = 'BayesNN'
    ROC_output_file_name = output_dir + '/ROC_' + method + '.svg'
    PR_output_file_name = output_dir + '/PR_' + method + '.svg'
    F1_output_file_name = output_dir + '/F1_' + method + '.svg'
    AUC = save_ROC_curve(label, scores_curves, method, ROC_output_file_name)
    save_PR_curve(label, scores_curves, method, PR_output_file_name, F1_output_file_name)
    confusion_mat = compute_confusion_matrix(label, scores_curves)
    ACC = accuracy_score(label, y_pred)

    results = {
        'score': vis_scores.data.cpu().numpy(),
        'Confusion Matrix': confusion_mat,
        'ACC': ACC,
        'AUC': AUC,
        'ROC': ROC_output_file_name,
        'PR': PR_output_file_name,
        'F1-score': F1_output_file_name
    }

    return results
