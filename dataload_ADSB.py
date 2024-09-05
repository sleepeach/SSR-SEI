from collections import Counter
import pandas as pd
import torch
import numpy as np
# from cleanlab.filter import find_label_issues
from sklearn.model_selection import train_test_split
import random


def Dataset():
    X = np.load("/data1/taomy/ADS-Bista-torch/ADS_B_4800/Task_1_Train_X_100Class.npy")
    X = X.transpose(2, 0, 1)
    Y = np.load("/data1/taomy/ADS-Bista-torch/ADS_B_4800/Task_1_Train_Y_100Class.npy")
    Y = Y.astype(np.uint8)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=30, stratify=Y)
    X_test = np.load("/data1/taomy/ADS-Bista-torch/ADS_B_4800/Task_1_Test_X_100Class.npy")
    X_test= X_test.transpose(2, 0, 1)
    Y_test = np.load("/data1/taomy/ADS-Bista-torch/ADS_B_4800/Task_1_Test_Y_100Class.npy")
    Y_test = Y_test.astype(np.uint8)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def Dataset_KshotRround(num, k, seed):
    X_train = np.load("/data1/taomy/ADS-Bista-torch/ADS_B_4800/Task_1_Train_X_100Class.npy")
    X_train = X_train.transpose(2, 0, 1)
    Y_train = np.load("/data1/taomy/ADS-Bista-torch/ADS_B_4800/Task_1_Train_Y_100Class.npy")
    Y_train = Y_train.astype(np.uint8)
    X_test = np.load("/data1/taomy/ADS-Bista-torch/ADS_B_4800/Task_1_Test_X_100Class.npy")
    X_test = X_test.transpose(2, 0, 1)
    Y_test = np.load("/data1/taomy/ADS-Bista-torch/ADS_B_4800/Task_1_Test_Y_100Class.npy")
    Y_test = Y_test.astype(np.uint8)
    X_train = np.append(X_train, X_test, axis=0)
    Y_train = np.append(Y_train, Y_test, axis=0)
    random_index_shot = []
    random.seed(seed)
    # random.seed(3)  # 如果固定种子的话，k+1shot是包含kshot里的样本
    for i in range(num):
        index_shot = [index for index, value in enumerate(Y_train) if value == i]
        random_index_shot += random.sample(index_shot, k)
    random.shuffle(random_index_shot)
    X_train_K_Shot = X_train[random_index_shot, :, :]
    Y_train_K_Shot = Y_train[random_index_shot]
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_train_K_Shot, Y_train_K_Shot, test_size=0.05,random_state=30, stratify=Y_train_K_Shot)
    return X_train_val, X_test, Y_train_val, Y_test


def wrongdataset(num, k, ratio, seed):
    X_train_all, X_test, Y_train_all, Y_test = Dataset_KshotRround(num, k,seed)
    z_all = torch.zeros(1900).numpy()
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


def asymmetric_wrongdataset(num, k, ratio, seed):
    X_train_all, X_test, Y_train_all, Y_test = Dataset_KshotRround(num, k,seed)
    z_all = torch.zeros(1900).numpy()
    n = int(Y_train_all.shape[0]* ratio)
    #wrong_num = int(ratio * Y_train.shape[0])  # 错误样本数
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
    #X_train, X_val, X_test, Y_train, Y_val, Y_test= Dataset()
    #X_train, X_test, Y_train, Y_test=Dataset_KshotRround(10, 200, 1000)
    X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, Y_train, Y_val, Y_test,Z_train, Z_val = asymmetric_wrongdataset(10, 200, 0.6,1000)
    #X_train, X_val, X_test, Y_train, Y_val, Y_test=wrongdataset(10, 200, 0.2)

    #X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, Y_train, Y_val, Y_test,Z_train, Z_val=wrongdataset(10, 200, 0,1000)
    #X_train, X_test, Y_train,  Y_test, Z_train = wrongdataset_all(10, 200, 0.2)

    #List= Y_train_K_Shot.tolist()
    # X = np.append(X_train, X_test, axis=0)
    # Y = np.append(Y_train, Y_test, axis=0)
    # X = np.append(X, X_val, axis=0)
    # Y = np.append(Y, Y_val, axis=0)
    #print(Y_train_all)
    print(Y_train_all)
    # print(Y_val)
    print(Y_test)
    a = Counter(Y_train_all)
    print(a)
    # b = Counter(Y_val)
    # print(b)
    c = Counter(Y_test)
    print(c)
    # d = Counter(Z_train)
    # print(d)
    # e = Counter(Z_val)
    # print(e)
    #print('X_train_all: ', X_train_all.shape[0])
    print('X_train: ', X_train.shape)
    #print('X_val: ', X_val.shape)
    #print('X_test: ', X_test.shape)
    #print('Y_train_all: ', Y_train_all.shape)
    print('Y_train: ', Y_train_all.shape)
    #print('Y_val: ', Y_val.shape)
    print('Y_test: ', Y_test.shape)
    print('Z_all: ', Z_all.shape)
    #print('Z_train: ', Z_train.shape)
    print('Z_train_1: ', torch.nonzero(torch.tensor(Z_all)).squeeze())
    print('Z_train_1_num: ', torch.nonzero(torch.tensor(Z_all)).squeeze().shape)
    # print('Z_val: ', Z_val.shape)
    # print('Z_val_1: ', torch.nonzero(Z_val).squeeze())
    # print('Z_val_1_num: ', torch.nonzero(Z_val).squeeze().shape)


