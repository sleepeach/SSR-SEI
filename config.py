import os
import torch
import random
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class ALL:
    seed = 1000
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

'''load dataset'''
class Dataset:
    data_type = "ADS-B"
    #data_type = "WiFi"
    #data_type = "LoRa"
    #noise_type = "asymmetric"
    noise_type = "symmetric"
    seed = ALL.seed
    device = ALL.device
    wrongratio = 0.2

    if data_type == "ADS-B":
        classes = 10
        number = 200
        batch_size = 32
        roc_auc_path = "roc_auc/ADS_B4800/CNN_batchsize%d_ratio%.2f_seed%d.png" % (batch_size, wrongratio, seed)
    if data_type == "WiFi":
        classes = 16
        number = 200
        batch_size = 32
        roc_auc_path = "roc_auc/WiFi/CNN_batchsize%d_ratio%.2f_seed%d.png" % (batch_size, wrongratio, seed)
    if data_type == "LoRa":
        classes = 15
        number = 250
        batch_size = 32
        roc_auc_path = "roc_auc/LoRa/CNN_batchsize%d_ratio%.2f_seed%d.png" % (batch_size, wrongratio, seed)


'''CrossEntropyLoss'''
class CE:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr: float = 0.001
    epoch = 300
    classes = Dataset.classes
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/logs_CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/ADS_B4800/CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            matrix_path = "matshow/ADS_B4800/CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.png" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            roc_auc_path ="roc_auc/ADS_B4800/CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.png" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            visualization_path="Visualization/ADS_B4800/CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.png" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)

        if data_type == "WiFi":
            writer = "logs/WiFi/logs_CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/WiFi/CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            matrix_path = "matshow/WiFi/CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.png" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            roc_auc_path = "roc_auc/WiFi/CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.png" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            visualization_path = "Visualization/WiFi/CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.png" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)

    if noise_type == "asymmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/asy_logs_CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/ADS_B4800/asy_CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
        if data_type == "WiFi":
            writer = "logs/WiFi/asy_logs_CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/WiFi/asy_CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)


'''mixup'''
class mixup:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr: float = 0.001
    epoch = 300
    classes = Dataset.classes
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/logs_CNN_mixup_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed)
            save_path = 'model_weight/ADS_B4800/CNN_mixup_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed)
        if data_type == "WiFi":
            writer = "logs/WiFi/logs_CNN_mixup_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/WiFi/CNN_mixup_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
    if noise_type == "asymmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/asy_logs_CNN_mixup_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/ADS_B4800/asy_CNN_mixup_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
        if data_type == "WiFi":
            writer = "logs/WiFi/asy_logs_CNN_mixup_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/WiFi/asy_CNN_mixup_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)


'''LabelSmoothingLoss'''
class LSR:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr = 0.001
    smoothing = 0.7
    epoch = 300
    classes = Dataset.classes
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/logs_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed,smoothing)
            save_path = 'model_weight/ADS_B4800/CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed,smoothing)
            visualization_path = "Visualization/ADS_B4800/CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.png" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
        if data_type == "WiFi":
            writer = "logs/WiFi/logs_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed,smoothing)
            save_path = 'model_weight/WiFi/CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed,smoothing)
            visualization_path = "Visualization/WiFi/CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.png" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)

    if noise_type == "asymmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/asy_logs_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed,smoothing)
            save_path = 'model_weight/ADS_B4800/asy_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed,smoothing)
        if data_type == "WiFi":
            writer = "logs/WiFi/asy_logs_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed,smoothing)
            save_path = 'model_weight/WiFi/asy_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed,smoothing)


'''Generalized_Cross_Entropy'''
class GCE:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr = 0.001
    r = 0.5
    epoch = 300
    classes = Dataset.classes
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/logs_CNN_GCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_r%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, r)
            save_path = 'model_weight/ADS_B4800/CNN_GCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_r%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, r)
        if data_type == "WiFi":
            writer = "logs/WiFi/logs_CNN_GCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_r%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, r)
            save_path = 'model_weight/WiFi/CNN_GCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_r%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, r)
    if noise_type == "asymmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/asy_logs_CNN_GCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_r%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, r)
            save_path = 'model_weight/ADS_B4800/asy_CNN_GCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_r%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, r)
        if data_type == "WiFi":
            writer = "logs/WiFi/asy_logs_CNN_GCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_r%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, r)
            save_path = 'model_weight/WiFi/asy_CNN_GCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_r%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, r)

'''Mean Absolute Error'''
class MAE:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr = 0.001
    scale = 2
    epoch = 300
    classes = Dataset.classes
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/logs_CNN_MAE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, scale)
            save_path = 'model_weight/ADS_B4800/CNN_MAE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, scale)
        if data_type == "WiFi":
            writer = "logs/WiFi/logs_CNN_MAE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, scale)
            save_path = 'model_weight/WiFi/CNN_MAE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, scale)
    if noise_type == "asymmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/asy_logs_CNN_MAE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, scale)
            save_path = 'model_weight/ADS_B4800/asy_CNN_MAE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, scale)
        if data_type == "WiFi":
            writer = "logs/WiFi/asy_logs_CNN_MAE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, scale)
            save_path = 'model_weight/WiFi/asy_CNN_MAE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, scale)


'''Symmetric_Cross_Entropy'''
class SCE:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr = 0.001
    alpha = 0.2
    beta = 0.8
    epoch = 300
    classes = Dataset.classes
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/logs_CNN_SCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, alpha, beta)
            save_path = 'model_weight/ADS_B4800/CNN_SCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, alpha, beta)
        if data_type == "WiFi":
            writer = "logs/WiFi/logs_CNN_SCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, alpha, beta)
            save_path = 'model_weight/WiFi/CNN_SCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, alpha, beta)
    if noise_type == "asymmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/asy_logs_CNN_SCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, alpha, beta)
            save_path = 'model_weight/ADS_B4800/asy_CNN_SCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, alpha, beta)
        if data_type == "WiFi":
            writer = "logs/WiFi/asy_logs_CNN_SCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, alpha, beta)
            save_path = 'model_weight/WiFi/asy_CNN_SCE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, alpha, beta)

'''metric'''
class metric:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr: float = 0.001
    epoch = 300
    num_classes = Dataset.classes
    feat_dim = 1024
    weight = 0.005
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/logs_CNN_metric_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed)
            save_path = 'model_weight/ADS_B4800/CNN_metric_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio,seed)
        if data_type == "WiFi":
            writer = "logs/WiFi/logs_CNN_metric_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/WiFi/CNN_metric_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
    if noise_type == "asymmetric":
        if data_type == "ADS-B":
            writer = "logs/ADS_B4800/asy_logs_CNN_metric_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/ADS_B4800/asy_CNN_metric_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
        if data_type == "WiFi":
            writer = "logs/WiFi/asy_logs_CNN_metric_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)
            save_path = 'model_weight/WiFi/asy_CNN_metric_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed)

'''SSR (Proposed)'''
class cleanlab_semi_supervised_LSR:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr: float = 0.001
    epoch1 = 16
    epoch = 300
    smoothing = 0.7
    m1 = 1
    m2 = 0.5
    weight = 0.1
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer1 = SummaryWriter("logs/ADS_B4800/logs_select_data_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing))
            save_path1 = 'model_weight/ADS_B4800/select_data_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            writer = SummaryWriter("logs/ADS_B4800/logs_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.1f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing))
            save_path = 'model_weight/ADS_B4800/cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            z_path1 = "select_sample/ADS_B4800/select_data1_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            z_path2 = "select_sample/ADS_B4800/select_data2_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            matrix_path = "matshow/ADS_B4800/CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.png" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)

        if data_type == "WiFi":
            writer1 = SummaryWriter("logs/WiFi/logs_select_data_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch1, lr, Dataset.wrongratio, seed, smoothing))
            save_path1 = 'model_weight/WiFi/select_data_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1, lr, Dataset.wrongratio, seed, smoothing)
            writer = SummaryWriter("logs/WiFi/logs_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, smoothing))
            save_path = 'model_weight/WiFi/cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, smoothing)
            z_path1 = "select_sample/WiFi/select_data1_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            z_path2 = "select_sample/WiFi/select_data2_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            matrix_path = "matshow/WiFi/CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.png" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)

    if noise_type == "asymmetric":
        if data_type == "ADS-B":
            writer1 = "logs/ADS_B4800/asy_logs_select_data_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            save_path1 = 'model_weight/ADS_B4800/sasy_elect_data_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            writer = "logs/ADS_B4800/asy_logs_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.1f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            save_path = 'model_weight/ADS_B4800/asy_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            z_path1 = "select_sample/ADS_B4800/asy_select_data1_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            z_path2 = "select_sample/ADS_B4800/asy_select_data2_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)

        if data_type == "WiFi":
            writer1 = "logs/WiFi/logs_select_data_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch1, lr, Dataset.wrongratio, seed, smoothing)
            save_path1 = 'model_weight/WiFi/select_data_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1, lr, Dataset.wrongratio, seed, smoothing)
            writer = "logs/WiFi/logs_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, smoothing)
            save_path = 'model_weight/WiFi/cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch, lr, Dataset.wrongratio, seed, smoothing)

    classes = Dataset.classes
    number = Dataset.number
    wrongratio = Dataset.wrongratio
    batch_size = Dataset.batch_size


'''Ablation Experiments (Effect of ASS)'''
class ASS_onetime:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr: float = 0.001
    epoch1 = 6
    epoch = 300
    smoothing = 0.7
    m1 = 1
    m2 = 0.5
    weight = 0.01
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer1 = "logs/ADS_B4800/logs_ASSonetime_select_data_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            save_path1 = 'model_weight/ADS_B4800/ASSonetime_select_data_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            writer = "logs/ADS_B4800/logs_ASSonetime_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.1f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            save_path = 'model_weight/ADS_B4800/ASSonetime_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing)
            z_path1 = "select_sample/ADS_B4800/select_data1_ASSonetime_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            z_path2 = "select_sample/ADS_B4800/select_data2_ASSonetime_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)

        if data_type == "WiFi":
            writer1 = "logs/WiFi/logs_ASSonetime_select_data_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            save_path1 = 'model_weight/WiFi/ASSonetime_select_data_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            writer = "logs/WiFi/logs_ASSonetime_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.1f" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            save_path = 'model_weight/WiFi/ASSonetime_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.pth' % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            z_path1 = "select_sample/WiFi/select_data1_ASSonetime_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)
            z_path2 = "select_sample/WiFi/select_data2_ASSonetime_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing)

    classes = Dataset.classes
    number = Dataset.number
    wrongratio = Dataset.wrongratio
    batch_size = Dataset.batch_size


'''Ablation Experiments (Effect of Dual Regularization)'''
class cleanlab_semi_supervised_otherloss:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    #loss_type = "CE"
    #loss_type = "GCE"
    loss_type = "SCE"
    lr: float = 0.001
    epoch1 = 16
    epoch = 300
    smoothing = 0.7
    r = 0.5
    alpha = 0.2
    beta = 0.8
    m1 = 1
    m2 = 0.5
    weight = 0.01
    classes = Dataset.classes
    number = Dataset.number
    wrongratio = Dataset.wrongratio
    batch_size = Dataset.batch_size
    if noise_type == "symmetric":
        if loss_type == "CE":
            if data_type == "ADS-B":
                writer1 = "logs/ADS_B4800/logs_select_data_cleanlab_CNN_otherlossCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d" % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                save_path1 = 'model_weight/ADS_B4800/select_data_cleanlab_Semi_Supervised_CNN_otherlossCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d.pth' % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                writer = "logs/ADS_B4800/logs_cleanlab_CNN_otherlossCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d" % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                save_path = 'model_weight/ADS_B4800/cleanlab_Semi_Supervised_CNN_otherlossCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d.pth' % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                z_path1 = "select_sample/ADS_B4800/select_data1_cleanlab_CNN_otherlossCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d.xlsx" % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                z_path2 = "select_sample/ADS_B4800/select_data2_cleanlab_CNN_otherlossCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d.xlsx" % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
            if data_type == "WiFi":
                writer1 = "logs/WiFi/logs_select_data_cleanlab_CNN_otherlossCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                save_path1 = 'model_weight/WiFi/select_data_cleanlab_Semi_Supervised_CNN_otherlossCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d.pth' % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                writer = "logs/WiFi/logs_cleanlab_CNN_otherlossCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                save_path = 'model_weight/WiFi/cleanlab_Semi_Supervised_CNN_otherlossCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d.pth' % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                z_path1 = "select_sample/WiFi/select_data1_cleanlab_CNN_otherlossCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d.xlsx" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
                z_path2 = "select_sample/WiFi/select_data2_cleanlab_CNN_otherlossCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d.xlsx" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed)
        if loss_type == "GCE":
            if data_type == "ADS-B":
                writer1 = "logs/ADS_B4800/logs_select_data_cleanlab_CNN_otherlossGCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f" % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                save_path1 = 'model_weight/ADS_B4800/select_data_cleanlab_Semi_Supervised_CNN_otherlossGCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f.pth' % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                writer = "logs/ADS_B4800/logs_cleanlab_CNN_otherlossGCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f" % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                save_path = 'model_weight/ADS_B4800/cleanlab_Semi_Supervised_CNN_otherlossGCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f.pth' % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                z_path1 = "select_sample/ADS_B4800/select_data1_cleanlab_CNN_otherlossGCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f.xlsx" % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                z_path2 = "select_sample/ADS_B4800/select_data2_cleanlab_CNN_otherlossGCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f.xlsx" % (
                Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
            if data_type == "WiFi":
                writer1 = "logs/WiFi/logs_select_data_cleanlab_CNN_otherlossGCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                save_path1 = 'model_weight/WiFi/select_data_cleanlab_Semi_Supervised_CNN_otherlossGCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f.pth' % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                writer = "logs/WiFi/logs_cleanlab_CNN_otherlossGCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                save_path = 'model_weight/WiFi/cleanlab_Semi_Supervised_CNN_otherlossGCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f.pth' % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                z_path1 = "select_sample/WiFi/select_data1_cleanlab_CNN_otherlossGCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f.xlsx" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
                z_path2 = "select_sample/WiFi/select_data2_cleanlab_CNN_otherlossGCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_r%.1f.xlsx" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, r)
        if loss_type == "SCE":
            if data_type == "ADS-B":
                writer1 = "logs/ADS_B4800/logs_select_data_cleanlab_CNN_otherlossSCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                save_path1 = 'model_weight/ADS_B4800/select_data_cleanlab_Semi_Supervised_CNN_otherlossSCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.pth' % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                writer = "logs/ADS_B4800/logs_cleanlab_CNN_otherlossSCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                save_path = 'model_weight/ADS_B4800/cleanlab_Semi_Supervised_CNN_otherlossSCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.pth' % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                z_path1 = "select_sample/ADS_B4800/select_data1_cleanlab_CNN_otherlossSCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.xlsx" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                z_path2 = "select_sample/ADS_B4800/select_data2_cleanlab_CNN_otherlossSCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.xlsx" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
            if data_type == "WiFi":
                writer1 = "logs/WiFi/logs_select_data_cleanlab_CNN_otherlossSCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                save_path1 = 'model_weight/WiFi/select_data_cleanlab_Semi_Supervised_CNN_otherlossSCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.pth' % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                writer = "logs/WiFi/logs_cleanlab_CNN_otherlossSCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                save_path = 'model_weight/WiFi/cleanlab_Semi_Supervised_CNN_otherlossSCE_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.pth' % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                z_path1 = "select_sample/WiFi/select_data1_cleanlab_CNN_otherlossSCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.xlsx" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)
                z_path2 = "select_sample/WiFi/select_data2_cleanlab_CNN_otherlossSCE_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_a%.2f_b%.2f.xlsx" % (
                    Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, alpha, beta)


'''Ablation Experiments (Effect of Dual Regularization)'''
class clean_CE:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    lr: float = 0.001
    epoch = 100
    writer = "logs/ADS_B4800/logs_cleanlab_CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d" %(Dataset.batch_size, epoch,lr,Dataset.wrongratio,seed)
    save_path ='model_weight/ADS_B4800/cleanlab_CNN_CE_batchsize%d_epoch%d_lr%.3f_ratio%.2f_seed%d.pth' %(Dataset.batch_size, epoch,lr,Dataset.wrongratio,seed)
    classes = Dataset.classes
    number = Dataset.number
    wrongratio = Dataset.wrongratio
    batch_size = Dataset.batch_size

'''SSR (Proposed)+sparse'''
class sparse_cleanlab_semi_supervised_LSR:
    seed = ALL.seed
    device = ALL.device
    data_type = Dataset.data_type
    noise_type = Dataset.noise_type
    lr: float = 0.001
    epoch1 = 16
    epoch = 300
    smoothing = 0.7
    alpha = 0.1
    m1 = 1
    m2 = 0.5
    weight = 0.01
    if noise_type == "symmetric":
        if data_type == "ADS-B":
            writer1 = SummaryWriter("logs/sparse_ADS_B4800/logs_select_data_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha))
            save_path1 = 'model_weight/sparse_ADS_B4800/select_data_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            writer = SummaryWriter("logs/sparse_ADS_B4800/logs_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha))
            save_path = 'model_weight/sparse_ADS_B4800/cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            sparse_save_path = 'model_weight/sparse_ADS_B4800/prune_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            z_path1 = "select_sample/sparse_ADS_B4800/select_data1_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing,alpha)
            z_path2 = "select_sample/sparse_ADS_B4800/select_data2_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing,alpha)
            matrix_path = "matshow/sparse_ADS_B4800/CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.png" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)

        if data_type == "WiFi":
            writer1 = SummaryWriter("logs/sparse_WiFi/logs_select_data_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha))
            save_path1 = 'model_weight/sparse_WiFi/select_data_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            writer = SummaryWriter("logs/sparse_WiFi/logs_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha))
            save_path = 'model_weight/sparse_WiFi/cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            sparse_save_path = 'model_weight/sparse_WiFi/prune_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            z_path1 = "select_sample/sparse_WiFi/select_data1_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing,alpha)
            z_path2 = "select_sample/sparse_WiFi/select_data2_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing,alpha)
            matrix_path = "matshow/sparse_WiFi/CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.png" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
        if data_type == "LoRa":
            writer1 = SummaryWriter("logs/sparse_LoRa/logs_select_data_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha))
            save_path1 = 'model_weight/sparse_LoRa/select_data_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            writer = SummaryWriter("logs/sparse_LoRa/logs_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha))
            save_path = 'model_weight/sparse_LoRa/cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            sparse_save_path = 'model_weight/sparse_LoRa/prune_cleanlab_Semi_Supervised_CNN_LSR_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.pth' % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)
            z_path1 = "select_sample/sparse_LoRa/select_data1_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing,alpha)
            z_path2 = "select_sample/sparse_LoRa/select_data2_cleanlab_CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.xlsx" % (Dataset.batch_size, epoch1, epoch, m1, m2, weight, lr, Dataset.wrongratio, seed, smoothing,alpha)
            matrix_path = "matshow/sparse_LoRa/CNN_LSR_Semi_Supervised_batchsize%d_preepoch%d_epoch%d_prem%.2f_m%.2f_weight_%.3f_lr%.3f_ratio%.2f_seed%d_smoth%.1f_alpha%.4f.png" % (Dataset.batch_size, epoch1,epoch,m1,m2,weight, lr, Dataset.wrongratio, seed,smoothing,alpha)

    classes = Dataset.classes
    number = Dataset.number
    wrongratio = Dataset.wrongratio
    batch_size = Dataset.batch_size

