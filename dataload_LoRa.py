from collections import Counter
import pandas as pd
import torch
import numpy as np
# from cleanlab.filter import find_label_issues
from sklearn.model_selection import train_test_split
import random
from numpy import sqrt
from numpy.random import standard_normal
import h5py

def awgn(data, snr_range):
    pkt_num = data.shape[0]
    SNRdB = random.uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        SNR_linear = 10**(SNRdB[pktIdx]/10)
        P = sum(abs(s)**2)/len(s)
        N0 = P/SNR_linear
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))
        data[pktIdx] = s + n

    return data

def convert_to_complex(data):
    '''Convert the loaded data to complex IQ samples.'''
    num_row = data.shape[0]
    num_col = data.shape[1]
    data_complex = np.zeros([num_row, 2, round(num_col/2)])
    data_complex[:,0,:] = data[:,:round(num_col/2)]
    data_complex[:,1,:] = data[:,round(num_col/2):]

    return data_complex[:,:,0:6000]

def LoadDataset(file_path, dev_range, pkt_range):
    '''
    Load IQ sample from a dataset
    Input:
    file_path is the dataset path
    dev_range specifies the loaded device range
    pkt_range specifies the loaded packets range

    Return:
    data is the loaded complex IQ samples
    label is the true label of each received packet
    '''

    dataset_name = 'data'
    labelset_name = 'label'

    f = h5py.File(file_path, 'r')
    label = f[labelset_name][:]
    label = label.astype(int)
    label = np.transpose(label)
    label = label - 1

    label_start = int(label[0]) + 1
    label_end = int(label[-1]) + 1
    num_dev = label_end - label_start + 1
    num_pkt = len(label)
    num_pkt_per_dev = int(num_pkt/num_dev)

    print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
          str(label_end) + ',' + str(num_pkt_per_dev) + ' packets per device.')

    sample_index_list = []

    for dev_idx in dev_range:
        sample_index_dev = np.where(label==dev_idx)[0][pkt_range]
        sample_index_list.extend(sample_index_dev)

    data = f[dataset_name][sample_index_list]
    data = convert_to_complex(data)
    label = label[sample_index_list]

    f.close()
    return data, label

def Get_LoRa_Dataset():
    file_path = '/data1/taomy/ADS-Bista-torch/LoRa_Dataset/Train/dataset_training_no_aug.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 500, dtype=int)
    x,y = LoadDataset(file_path, dev_range, pkt_range)
    y = y.astype(np.uint8)
    y = y.flatten()
    # X_train_val, X_test, Y_train_val, Y_test = train_test_split(x, y, test_size=0.1, random_state=30)
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1, random_state=30)
    # return X_train, X_val, X_test, Y_train, Y_val, Y_test
    return x,y

def Dataset_KshotRround(num, k, seed):
    X_train, Y_train = Get_LoRa_Dataset()
    random_index_shot = []
    random.seed(seed)
    for i in range(num):
        index_shot = [index for index, value in enumerate(Y_train) if value == i]
        random_index_shot += random.sample(index_shot, k)
    random.shuffle(random_index_shot)
    X_train_K_Shot = X_train[random_index_shot, :, :]
    Y_train_K_Shot = Y_train[random_index_shot]
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_train_K_Shot, Y_train_K_Shot, test_size=0.05,random_state=30, stratify=Y_train_K_Shot)
    return X_train_val, X_test, Y_train_val, Y_test


def LoRa_wrongdataset(num, k, ratio, seed):
    X_train_all, X_test, Y_train_all, Y_test = Dataset_KshotRround(num, k, seed)
    z_all = torch.zeros(Y_train_all.shape[0]).numpy()
    wrong_num = int(ratio * Y_train_all.shape[0])
    index = list(range(Y_train_all.shape[0]))
    random.seed(seed)
    wrong_index = random.sample(index, wrong_num)
    for i in wrong_index:
        remain_class = [m for m in range(num) if m != Y_train_all[i]]
        Y_train_all[i] = random.choice(remain_class)
        z_all[i] = 1
    X_train, X_val, Y_train, Y_val, Z_train, Z_val = train_test_split(X_train_all, Y_train_all, z_all, test_size=0.1,random_state=30, stratify=Y_train_all)
    return X_train_all, Y_train_all, z_all, X_train, X_val, X_test, Y_train, Y_val, Y_test, Z_train, Z_val

if __name__ == '__main__':
    # x,y = Get_LoRa_Dataset()
    # print(x.shape)
    # print(y.shape)
    # print(Counter(y))
    #X_train, X_test, Y_train, Y_test=Dataset_KshotRround(30, 200, 1000)

    X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, Y_train, Y_val, Y_test,Z_train, Z_val=LoRa_wrongdataset(15, 250, 0.2,1000)

    a = Counter(Y_train_all)
    print(a)
    b = Counter(Y_val)
    print(b)
    c = Counter(Y_test)
    print(c)
    d = Counter(Z_train)
    print(d)
    e = Counter(Z_val)
    print(e)
    print('X_train_all: ', X_train_all.shape[0])
    #print('X_train: ', X_train.shape)
    print('X_val: ', X_val.shape)
    print('X_test: ', X_test.shape)
    print('Y_train_all: ', Y_train_all.shape)
    #print('Y_train: ', Y_train)
    print('Y_val: ', Y_val.shape)
    print('Y_test: ', Y_test.shape)
    print('Z_all: ', Z_all.shape)
    #print('Z_train: ', Z_train.shape)
    print('Z_train_1: ', torch.nonzero(torch.tensor(Z_all)).squeeze())
    print('Z_train_1_num: ', torch.nonzero(torch.tensor(Z_all)).squeeze().shape)
    # print('Z_val: ', Z_val.shape)
    # print('Z_val_1: ', torch.nonzero(Z_val).squeeze())
    # print('Z_val_1_num: ', torch.nonzero(Z_val).squeeze().shape)


