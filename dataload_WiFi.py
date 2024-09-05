import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import train_test_split
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform
from scipy import signal
import math
import json
import h5py
from collections import Counter


def convert_to_I_Q_complex(data):
    '''Convert the loaded data to complex I and Q samples.'''
    num_row = data.shape[0]
    num_col = data.shape[1]
    data_complex = np.zeros([num_row, 2, round(num_col/2)])
    data_complex[:,0,:] = data[:,:round(num_col/2)]
    data_complex[:,1,:] = data[:,round(num_col/2):]

    return data_complex


def Power_Normalization(x):
    for i in range(x.shape[0]):
        max_power = (np.power(x[i,0,:],2) + np.power(x[i,1,:],2)).max()
        x[i] = x[i] / np.power(max_power, 1/2)
    return x

def WiFi_Dataset_slice(ft):
    devicename = ['3123D7B', '3123D7D', '3123D7E', '3123D52', '3123D54', '3123D58', '3123D64', '3123D65',
                  '3123D70', '3123D76', '3123D78', '3123D79', '3123D80', '3123D89', '3123EFE', '3124E4A']
    data_IQ_wifi_all = np.zeros((1,2,6000))
    data_target_all = np.zeros((1,))
    target = 0
    for classes in range(16):
        for recoder in range(1):
            inputFilename = f'/data1/taomy/ADS-Bista-torch/WiFi_real/KRI-16Devices-RawData/{ft}ft/WiFi_air_X310_{devicename[classes]}_{ft}ft_run{recoder+1}'
            with open("{}.sigmf-meta".format(inputFilename),'rb') as read_file:
                meta_dict = json.load(read_file)
            with open("{}.sigmf-data".format(inputFilename),'rb') as read_file:
                binary_data = read_file.read()
            fullVect = np.frombuffer(binary_data, dtype=np.complex128)
            even = np.real(fullVect) #提取复数信号中的实部
            odd = np.imag(fullVect)  #提取复数信号中的虚部
            length = 6000
            num = 0
            data_IQ_wifi = np.zeros((math.floor(len(even)/length), 2, 6000))
            data_target = np.zeros((math.floor(len(even)/length),))
            for begin in range(0,len(even)-(len(even)-math.floor(len(even)/length)*length),length):
                data_IQ_wifi[num,0,:] = even[begin:begin+length]
                data_IQ_wifi[num,1,:] = odd[begin:begin+length]
                data_target[num,] = target
                num = num + 1
            data_IQ_wifi_all = np.concatenate((data_IQ_wifi_all,data_IQ_wifi),axis=0)
            data_target_all = np.concatenate((data_target_all, data_target), axis=0)
        target = target + 1

    return data_IQ_wifi_all[1:,], data_target_all[1:,]


def task_IL(ft):
    x, y = WiFi_Dataset_slice(ft)
    x = Power_Normalization(x)
    y = y.astype(np.uint8)
    # X_train_1, X_train_2, Y_train_1, Y_train_2 = train_test_split(x, y, test_size=0.3, random_state=30)
    #X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=30)
    # np.save(f'./WiFi_newdata/X_pretrain.npy', X_train_2)
    # np.save(f'./WiFi_newdata/Y_pretrain.npy', Y_train_2)
    # np.save(f'./WiFi_newdata/X_test.npy', X_test)
    # np.save(f'./WiFi_newdata/Y_test.npy', Y_test)
    # return X_train, X_test, Y_train, Y_test
    return x,y

def Dataset_KshotRround(num, k, seed, ft):
    X_train, Y_train = task_IL(ft)
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


def WiFi_wrongdataset(num, k, ratio, seed, ft):
    X_train_all, X_test, Y_train_all, Y_test = Dataset_KshotRround(num, k, seed, ft)
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


def WiFi_asymmetric_wrongdataset(num, k, ratio, seed, ft):
    X_train_all, X_test, Y_train_all, Y_test = Dataset_KshotRround(num, k,seed, ft)
    z_all = torch.zeros(Y_train_all.shape[0]).numpy()
    wrong_num = int(Y_train_all.shape[0]* ratio)
    index = list(range(Y_train_all.shape[0]))  # 样本序号列表
    #wrong_index = random.sample(index, n)  # 错误的样本的序号
    for i in index:
        if ratio == 0:
            Y_train_all[i] = str(Y_train_all[i])
        if ratio == 0.1:
            if Y_train_all[i] == 0:
                Y_train_all[i] = str(Y_train_all[i]).replace('0', '9')
                z_all[i] = 1
        if ratio == 0.2:
            if Y_train_all[i] == 0:
                Y_train_all[i] = str(Y_train_all[i]).replace('0', '9')
                z_all[i] = 1
            if Y_train_all[i] == 1:
                Y_train_all[i] = str(Y_train_all[i]).replace('1', '9')
                z_all[i] = 1
        if ratio == 0.4:
            if Y_train_all[i] == 0:
                Y_train_all[i] = str(Y_train_all[i]).replace('0', '9')
                z_all[i] = 1
            if Y_train_all[i] == 1:
                Y_train_all[i] = str(Y_train_all[i]).replace('1', '9')
                z_all[i] = 1
            if Y_train_all[i] == 2:
                Y_train_all[i] = str(Y_train_all[i]).replace('2', '9')
                z_all[i] = 1
            if Y_train_all[i] == 3:
                Y_train_all[i] = str(Y_train_all[i]).replace('3', '9')
                z_all[i] = 1
        elif ratio == 0.6:
            if Y_train_all[i] == 0:
                Y_train_all[i] = str(Y_train_all[i]).replace('0', '9')
                z_all[i] = 1
            if Y_train_all[i] == 1:
                Y_train_all[i] = str(Y_train_all[i]).replace('1', '9')
                z_all[i] = 1
            if Y_train_all[i] == 2:
                Y_train_all[i] = str(Y_train_all[i]).replace('2', '9')
                z_all[i] = 1
            if Y_train_all[i] == 3:
                Y_train_all[i] = str(Y_train_all[i]).replace('3', '9')
                z_all[i] = 1
            if Y_train_all[i] == 4:
                Y_train_all[i] = str(Y_train_all[i]).replace('4', '9')
                z_all[i] = 1
            if Y_train_all[i] == 5:
                Y_train_all[i] = str(Y_train_all[i]).replace('5', '9')
                z_all[i] = 1

    X_train, X_val, Y_train, Y_val, Z_train, Z_val = train_test_split(X_train_all, Y_train_all, z_all, test_size=0.1,random_state=30, stratify=Y_train_all)
    return X_train_all, Y_train_all, z_all, X_train, X_val, X_test, Y_train, Y_val, Y_test, Z_train, Z_val




if __name__ == '__main__':
    ft = 62
    #X_train, X_test, Y_train, Y_test = task_IL(ft)
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    #WiFi_IL100(ft)
    # print(X_train)
    # print(Y_train)
    # X = np.load("E:\pythoncode\未知信号_145tmy+yzy\WiFi_newdata\X_train_split_1.npy")
    # Y = np.load("E:\pythoncode\未知信号_145tmy+yzy\WiFi_newdata\Y_train_split_1.npy")
    # # # X = np.load("E:\pythoncode\未知信号_145tmy+yzy\WiFi_newdata\X_test.npy")
    # # # Y = np.load("E:\pythoncode\未知信号_145tmy+yzy\WiFi_newdata\Y_test.npy")
    # X, Y = task_IL(ft)

    # print(X.shape)
    # print(Y.shape)
    # print(X)
    # print(Y)
    # print(Counter(Y))
    # X_train, X_test, Y_train, Y_test = Dataset_KshotRround(16, 300, 1000, ft)
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    # print(Y_train)
    # print(Y_test)
    # print(Counter(Y_train))
    # print(Counter(Y_test))
    X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, Y_train, Y_val, Y_test, Z_train, Z_val = WiFi_asymmetric_wrongdataset(16, 3000, 0.2, 1000, ft)
    print(Y_train_all)
    print(Y_test)
    a = Counter(Y_train_all)
    print(a)
    c = Counter(Y_test)
    print(c)
    print('Y_train: ', Y_train_all.shape)
    print('Y_test: ', Y_test.shape)
    print('Z_all: ', Z_all.shape)
    print('Z_train_1: ', torch.nonzero(torch.tensor(Z_all)).squeeze())
    print('Z_train_1_num: ', torch.nonzero(torch.tensor(Z_all)).squeeze().shape)
