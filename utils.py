import math
import random
import os
import numpy as np
import pandas as pd
import torch
from shutil import copyfile
from cleanlab.filter import find_label_issues
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from thop import profile
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.patheffects as PathEffects
from typing import List
import mat73
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from dataload_ADSB import wrongdataset, asymmetric_wrongdataset
from dataload_LoRa import LoRa_wrongdataset
from dataload_WiFi import WiFi_wrongdataset
from model_complexcnn_onlycnn import base_complex_model

'''设置随机种子'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


'''归一化与训练数据'''
def Data_prepared(HP):
    if HP.noise_type == "symmetric":
        if HP.data_type == "ADS-B":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed)
        if HP.data_type == "WiFi":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = WiFi_wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed, 62)
        if HP.data_type == "LoRa":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = LoRa_wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed)
    if HP.noise_type == "asymmetric":
        if HP.data_type == "ADS-B":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = asymmetric_wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed)
        if HP.data_type == "WiFi":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = WiFi_wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed, 62)

    '''归一化'''
    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val
    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    X_train_all = (X_train_all - min_value) / (max_value - min_value)
    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(value_Y_train), torch.Tensor(Z_train))
    train_dataloader = DataLoader(train_dataset, batch_size=HP.batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val), torch.Tensor(Z_val))
    val_dataloader = DataLoader(val_dataset, batch_size=HP.batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=HP.batch_size, shuffle=True)
    all_train_dataset = TensorDataset(torch.Tensor(X_train_all), torch.Tensor(Y_train_all), torch.Tensor(Z_all))
    all_train_dataloader = DataLoader(all_train_dataset, batch_size=HP.batch_size, shuffle=True)

    return all_train_dataloader,train_dataloader, val_dataloader, test_dataloader


'''训练模型过程'''
def train(model, loss, train_dataloader, optimizer, epoch, writer, device):
    model.train()
    correct = 0
    train_loss = 0
    for data, target,z in train_dataloader:
        target = target.long()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss_batch = loss(output[1], target)
        #train_loss_batch_ = train_loss_batch / data.shape[0]
        #train_loss_batch_.backward()
        train_loss_batch.backward()
        optimizer.step()
        train_loss += train_loss_batch.item()
        pred = output[1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_dataloader.dataset)
    print('Train Epoch: {} \nTrain set: Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        train_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', train_loss, epoch)

'''验证模型过程'''
def val(model, loss, val_dataloader, epoch, writer, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target,z in val_dataloader:
            target = target.long()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            #output = F.log_softmax(output[1], dim=1)
            val_loss += loss(output[1], target).item()
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_dataloader.dataset)
    fmt = 'Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            val_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/validation', 100.0 * correct / len(val_dataloader.dataset), epoch)
    writer.add_scalar('Loss/validation', val_loss,epoch)
    return val_loss,100.0 * correct / len(val_dataloader.dataset)

'''训练与验证模型过程'''
def train_and_val(model, loss_function, train_dataloader, val_dataloader, optimizer, HP):
    current_min_val_loss = 100
    for epoch in range(1, HP.epoch + 1):
        train(model, loss_function,  train_dataloader, optimizer, epoch, HP.writer, HP.device)
        val_loss,val_correct = val(model, loss_function, val_dataloader, epoch, HP.writer, HP.device)
        if val_loss < current_min_val_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_val_loss, val_loss))
            current_min_val_loss = val_loss
            torch.save(model, HP.save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")


'''测验模型过程'''
def evaluate(model,test_dataloader,HP):
    model.eval()
    model = model.to(HP.device)
    correct = 0
    target_pred =[]
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(HP.device)
            target = target.to(HP.device)
            output = model(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            target_pred[len(target_pred):len(target) - 1] = pred.tolist()
            target_real[len(target_real):len(target) - 1] = target.tolist()
        target_pred = np.array(target_pred)
        target_real = np.array(target_real)
    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    print(precision_score(target_real, target_pred, average='macro'))
    print(recall_score(target_real, target_pred, average='macro'))
    print(f1_score(target_real, target_pred, average='macro'))

'''训练sparse模型过程'''
def sparse_train(model, loss, train_dataloader, optimizer1, optimizer2, epoch, writer, device,alpha):
    model.train()
    correct = 0
    train_loss = 0
    r1_loss = 0
    for data, target, z in train_dataloader:
        target = target.long()
        data = data.to(device)
        target = target.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        output = model(data)

        zero_data = torch.zeros(model.lamda.size()).to(device)
        r1_loss_batch = F.l1_loss(model.lamda, zero_data, reduction='sum')

        train_loss_batch = loss(output[1], target) + alpha * r1_loss_batch
        if alpha == 0:
            model.lamda.requires_grad_(False)
        else:
            model.lamda.requires_grad_(True)
        train_loss_batch.backward()
        optimizer1.step()
        optimizer2.step()
        train_loss += train_loss_batch.item()
        r1_loss += r1_loss_batch.item()
        pred = output[1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_dataloader.dataset)
    print('Train Epoch: {} \nTrain set: Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        train_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('R1_Loss/train', r1_loss, epoch)

'''训练与验证模型过程'''
def sparse_train_and_val(model, loss_function, train_dataloader, val_dataloader, optimizer1, optimizer2, HP):
    current_min_val_loss = 100
    for epoch in range(1, HP.epoch + 1):
        sparse_train(model, loss_function,  train_dataloader, optimizer1,optimizer2, epoch, HP.writer, HP.device,HP.alpha)
        val_loss,val_correct = val(model, loss_function, val_dataloader, epoch, HP.writer, HP.device)

        if val_loss < current_min_val_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_val_loss, val_loss))
            current_min_val_loss = val_loss
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
    torch.save(model, HP.save_path)

'''
fuction: 修剪对应行
'''
def prune_new(input1,input2,HP):
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()
    i = np.nonzero(input1)
    input2 = np.array(input2)
    input2_new = input2[i, :]
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    input2_new_ = input2_new_.to(HP.device)
    input2_new_ = input2_new_.squeeze()
    return input2_new_
'''
fuction: 修剪对应列
'''
def prune_new2(input1,input2,HP):
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()
    i = np.nonzero(input1)
    input2 = np.array(input2)
    input2_new = input2[:, i]
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    input2_new_ = input2_new_.to(HP.device)
    input2_new_ = input2_new_.squeeze()
    return input2_new_
'''
fuction: 修剪对应元素
'''
def prune_new3(input1,input2,HP):
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()
    i = np.nonzero(input1)
    input2 = np.array(input2)
    input2_new = input2[i]
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    input2_new_ = input2_new_.to(HP.device)
    input2_new_ = input2_new_.squeeze()
    return input2_new_

'''
fuction: 模型剪枝并计算相应指标
'''
def prune_model_and_test(model_prune,HP,test_dataloader):
    # savepath已保存的模型
    # loadpath剪枝后模型
    save_path = HP.save_path
    sparse_save_path = HP.sparse_save_path
    model = torch.load(save_path)
    torch.save(model.state_dict(), sparse_save_path)
    dict = torch.load(sparse_save_path)
    # 进行剪枝
    tensor_new = prune_new(dict["lamda"], dict["linear1.weight"], HP)
    dict["linear1.weight"] = tensor_new
    tensor_new2 = prune_new2(dict["lamda"], dict["linear2.weight"], HP)
    dict["linear2.weight"] = tensor_new2
    tensor_new3 = prune_new3(dict["lamda"], dict["linear1.bias"], HP)
    dict["linear1.bias"] = tensor_new3
    tensor_lamda = prune_new3(dict["lamda"], dict["lamda"], HP)
    dict["lamda"] = tensor_lamda
    # 计算剪枝后linear层的计算量与参数量
    params_linear = dict["linear1.weight"].size()[0] * (dict["linear1.weight"].size()[1] + 1) + \
                    dict["linear2.weight"].size()[0] * (dict["linear2.weight"].size()[1] + 1)
    # flops_linear = 2*model.state_dict()["linear1.weight"].size()[0]*model.state_dict()["linear1.weight"].size()[1]+2*model.state_dict()["linear2.weight"].size()[0]*model.state_dict()["linear2.weight"].size()[1]
    flops_linear = dict["linear1.weight"].size()[0] * (dict["linear1.weight"].size()[1] + 1) + \
                   dict["linear2.weight"].size()[0] * (dict["linear2.weight"].size()[1] + 1)
    # 计算特征稀疏度
    m = 0
    for i in tensor_lamda.cpu().numpy():
        if i != 0:
            m = m + 1
    print('特征维度:', str(m))
    print('特征稀疏度:', str(m / 1024))
    print(tensor_lamda.cpu().numpy())
    # 保存剪枝后的新模型
    model_new = model_prune(m)
    model_new.load_state_dict(dict)
    model_new = model_new.to(HP.device)
    torch.save(model_new, sparse_save_path)
    # 计算新模型的参数量与计算量
    input = torch.randn((1, 2, 6000))
    flops, params = profile(model_new.cpu(), inputs=(input,))
    print('flops:', str((flops + flops_linear) / 1000 ** 3) + " " + 'G' )
    print('params:',str((params + params_linear) / 1000 ** 2) + " " + 'M')
    # 测试相关指标
    model_new = model_new.to(HP.device)
    model_new.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(HP.device)
            target = target.to(HP.device)
            output = model_new(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

'''测验模型过程'''
def sparse_evaluate(model,test_dataloader,HP):
    model.eval()
    model = model.to(HP.device)
    correct = 0
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(HP.device)
            target = target.to(HP.device)
            output = model(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            target_pred[len(target_pred):len(target) - 1] = pred.tolist()
            target_real[len(target_real):len(target) - 1] = target.tolist()
        target_pred = np.array(target_pred)
        target_real = np.array(target_real)
    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    print(precision_score(target_real, target_pred, average='macro'))
    print(recall_score(target_real, target_pred, average='macro'))
    print(f1_score(target_real, target_pred, average='macro'))
    j = 0
    for i in model.lamda.cpu().detach().numpy():
        if i != 0:
            j = j + 1
    print('number_lamda!=0: \t', j)

'''mixup'''
def mixup_train(model, loss, train_dataloader, optimizer, epoch, writer, device):
    model.train()
    correct = 0
    train_loss = 0
    for data, target,z in train_dataloader:
        target = target.long()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        # Mixup
        lam = np.random.beta(1, 1)
        index = torch.randperm(data.size(0)).to(device)
        inputs = lam * data + (1 - lam) * data[index, :]
        target_a, target_b = target, target[index]
        optimizer.zero_grad()
        output = model(inputs)
        train_loss_batch = lam * loss(output[1], target_a) + (1 - lam) * loss(output[1], target_b)
        train_loss_batch.backward()
        optimizer.step()
        train_loss += train_loss_batch.item()
        pred = output[1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_dataloader.dataset)
    print('Train Epoch: {} \nTrain set: Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        train_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer = SummaryWriter(writer)
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', train_loss, epoch)


def mixup_train_and_val(model, loss_function, train_dataloader, val_dataloader, optimizer, HP):
    current_min_val_loss = 100
    for epoch in range(1, HP.epoch + 1):
        mixup_train(model, loss_function,  train_dataloader, optimizer, epoch, HP.writer, HP.device)
        val_loss,val_correct = val(model, loss_function, val_dataloader, epoch, HP.writer, HP.device)
        if val_loss < current_min_val_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_val_loss, val_loss))
            current_min_val_loss = val_loss
            torch.save(model, HP.save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")


'''centerloss'''
def metric_learning_train(model, loss1,loss2, train_dataloader, optimizer1, optimizer2, weight, epoch, writer, device):
    model.train()
    correct = 0
    train_loss = 0
    for data, target,z in train_dataloader:
        target = target.long()
        data = data.to(device)
        target = target.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        output = model(data)
        ce_loss = loss1(output[1], target)
        center_loss = loss2(output[0], target)
        train_loss_batch = ce_loss + weight * center_loss
        train_loss_batch.backward()
        optimizer1.step()
        for param in loss2.parameters():
            param.grad.data *= (1. / weight)
        optimizer2.step()
        train_loss += train_loss_batch.item()
        pred = output[1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_dataloader.dataset)
    print('Train Epoch: {} \nTrain set: Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        train_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer = SummaryWriter(writer)
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', train_loss, epoch)

def metric_learning_train_and_val(model, loss_function1,loss_function2, train_dataloader, val_dataloader, optimizer1,optimizer2, HP):
    current_min_val_loss = 100
    for epoch in range(1, HP.epoch + 1):
        metric_learning_train(model, loss_function1, loss_function2, train_dataloader, optimizer1,optimizer2, HP.weight, epoch, HP.writer, HP.device)
        val_loss,val_correct = val(model, loss_function1, val_dataloader, epoch, HP.writer, HP.device)
        if val_loss < current_min_val_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_val_loss, val_loss))
            current_min_val_loss = val_loss
            torch.save(model, HP.save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")


'''cleanlab+Semi-Supervised Learning'''
def cleanlab_Semi_Supervised_Data_prepared(model,HP,m,z_path):
    if HP.noise_type == "symmetric":
        if HP.data_type == "ADS-B":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed)
        if HP.data_type == "WiFi":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = WiFi_wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed, 62)
        if HP.data_type == "LoRa":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = LoRa_wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed)

    if HP.noise_type == "asymmetric":
        if HP.data_type == "ADS-B":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = asymmetric_wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed)
        if HP.data_type == "WiFi":
            X_train_all, Y_train_all, Z_all, X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val = WiFi_wrongdataset(HP.classes, HP.number, HP.wrongratio, HP.seed, 62)

    min_value = X_train_all.min()
    max_value = X_train_all.max()
    X_train = (X_train_all - min_value) / (max_value - min_value)
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train_all), torch.Tensor(Z_all))
    train_dataloader = DataLoader(train_dataset, batch_size=HP.batch_size, shuffle=False)

    pred_prob_ =[]
    target_real_ =[]
    model.eval()
    i = 0
    with torch.no_grad():
        for data, target, z in train_dataloader:
            data = data.to(HP.device)
            target = target.long().to(HP.device)
            output = model(data)
            pred = F.softmax(output[1], dim=1)
            pred_prob = np.array(pred.cpu().numpy())
            target_real = np.array(target.cpu().numpy()).T.squeeze()
            pred_prob_.extend(pred_prob)
            target_real_.extend(target_real)

    target_real_ = np.array(target_real_)
    pred_prob_ = np.array(pred_prob_)

    confidence_thresholds = calculate_confidence_threshold(target_real_, pred_prob_, HP.classes)
    target_real_ = torch.Tensor(target_real_).long()
    pred_prob_ = torch.Tensor(pred_prob_)
    incorrect_index = filter_incorrect_samples(target_real_, pred_prob_, confidence_thresholds)

    ordered_label_errors = incorrect_index
    print(len(ordered_label_errors))
    '''将Z_all[ordered_label_errors]存下来'''
    data_z = pd.DataFrame(np.array(Z_all[ordered_label_errors]))
    writer = pd.ExcelWriter(z_path)
    data_z.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer.close()

    ordered_label_errors_half = round(len(ordered_label_errors) * m)
    ordered_label_errors = ordered_label_errors[:ordered_label_errors_half]
    print(len(ordered_label_errors))

    print('Z_train_1_num: ', torch.count_nonzero(torch.tensor(Z_all[ordered_label_errors]).squeeze()))

    right_order = [num for num, i in enumerate(Y_train_all) if num not in ordered_label_errors]
    X_train_labeled = X_train[right_order]
    value_Y_train = Y_train_all[right_order]
    Z_train = Z_all[right_order]
    X_train_unlabeled = X_train[ordered_label_errors]
    return X_train_labeled, X_train_unlabeled, X_val, X_test, value_Y_train, value_Y_val, Y_test, Z_train, Z_val


'''cleanlab_Semi_Supervised_训练模型过程'''
def cleanlab_Semi_Supervised_train(model, loss, train_labeled_dataloader, train_unlabeled_dataloader, optimizer, epoch, writer, device,weight):
    model.train()
    correct = 0
    train_loss = 0
    for (traindata_labeled, traindata_unlabeled) in zip(train_labeled_dataloader, train_unlabeled_dataloader):
        data_labeled, target, z = traindata_labeled
        data_unlabeled, = traindata_unlabeled
        target = target.long()
        data_labeled = data_labeled.to(device)
        data_unlabeled = data_unlabeled.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output_labeled = model(data_labeled)
        output_unlabeled = model(data_unlabeled)
        output_unlabeled_ = F.softmax(output_unlabeled[1], dim=1)
        EM = (-1 * torch.sum(output_unlabeled_ * torch.log(output_unlabeled_), dim=1)).sum()
        train_loss_batch = loss(output_labeled[1], target) + weight * EM
        #train_loss_batch = loss(output_labeled[1], target) + loss(output_unlabeled_, output_unlabeled_)
        #train_loss_batch_ = train_loss_batch / data.shape[0]
        #train_loss_batch_.backward()
        train_loss_batch.backward()
        optimizer.step()
        train_loss += train_loss_batch.item()
        pred = output_labeled[1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_labeled_dataloader.dataset)
    print('Train Epoch: {} \nTrain set: Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        train_loss,
        correct,
        len(train_labeled_dataloader.dataset),
        100.0 * correct / len(train_labeled_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_labeled_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', train_loss, epoch)

def cleanlab_Semi_Supervised_train_and_val(model, loss_function0, loss_function1, train_dataloader_orignal, val_dataloader_orignal, optimizer, HP):
    current_min_val_loss = 100
    i = 0
    m = HP.m1
    z_path = HP.z_path1
    train_labeled_dataloader_ = []
    train_unlabeled_dataloader_ = []
    val_dataloader_ = []
    train_labeled_dataloader = []
    val_dataloader = []
    for epoch in range(1, HP.epoch1 + 1):
        if epoch <= 5:
            train_dataloader = train_dataloader_orignal
            val_dataloader = val_dataloader_orignal
            train(model, loss_function0, train_dataloader, optimizer, epoch, HP.writer1, HP.device)
            val_loss, val_correct = val(model, loss_function0, val_dataloader, epoch, HP.writer1, HP.device)
            if val_loss < current_min_val_loss:
                print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                    current_min_val_loss, val_loss))
                current_min_val_loss = val_loss
            else:
                print("The validation loss is not improved.")
            print("------------------------------------------------")
        else:
            if i % 10 == 0:
                X_train_labeled, X_train_unlabeled, X_val, X_test, Y_train, Y_val, Y_test, Z_train, Z_val = cleanlab_Semi_Supervised_Data_prepared(model, HP, m,z_path)
                m = HP.m2
                z_path = HP.z_path2
                k = round(HP.batch_size * X_train_unlabeled.shape[0] / X_train_labeled.shape[0])
                train_labeled_dataset = TensorDataset(torch.Tensor(X_train_labeled),torch.Tensor(Y_train),torch.Tensor(Z_train))
                train_labeled_dataloader = DataLoader(train_labeled_dataset, batch_size=HP.batch_size, shuffle=True)

                train_unlabeled_dataset = TensorDataset(torch.Tensor(X_train_unlabeled))
                train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, batch_size=k, shuffle=True)

                X_train_later, X_val_later, Y_train_later, Y_val_later, Z_train_later, Z_val_later = train_test_split(X_train_labeled, Y_train, Z_train, test_size=0.1, random_state=30,stratify=Y_train)
                #X_train_later, X_val_later, Y_train_later, Y_val_later, Z_train_later, Z_val_later = train_test_split(X_train_labeled, Y_train, Z_train, test_size=0.1, random_state=30)

                val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val), torch.Tensor(Z_val))
                val_dataloader = DataLoader(val_dataset, batch_size=HP.batch_size, shuffle=True)

                train_dataset_later = TensorDataset(torch.Tensor(X_train_later),torch.Tensor(Y_train_later),torch.Tensor(Z_train_later))
                train_dataloader_later = DataLoader(train_dataset_later, batch_size=HP.batch_size, shuffle=True)

                val_dataset_later = TensorDataset(torch.Tensor(X_val_later), torch.Tensor(Y_val_later), torch.Tensor(Z_val_later))
                val_dataloader_later = DataLoader(val_dataset_later, batch_size=HP.batch_size, shuffle=True)

                train_labeled_dataloader_ = train_labeled_dataloader
                train_unlabeled_dataloader_ = train_unlabeled_dataloader
                val_dataloader_ = val_dataloader
            else:
                train_labeled_dataloader = train_labeled_dataloader_
                train_unlabeled_dataloader = train_unlabeled_dataloader_
                val_dataloader = val_dataloader_
            cleanlab_Semi_Supervised_train(model, loss_function1, train_labeled_dataloader, train_unlabeled_dataloader,optimizer, epoch, HP.writer1, HP.device, HP.weight)
            val_loss, val_correct = val(model, loss_function1, val_dataloader, epoch, HP.writer1, HP.device)
            if val_loss < current_min_val_loss:
                print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                    current_min_val_loss, val_loss))
                current_min_val_loss = val_loss
                torch.save(model, HP.save_path)
            else:
                print("The validation loss is not improved.")
            print("------------------------------------------------")
            i = i + 1
    return train_dataloader_later, val_dataloader_later

'''画混淆矩阵'''
def confusion_matrix_mat(model, test_dataloader, HP):
    model.eval()
    model = model.to(HP.device)
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(HP.device)
            target = target.to(HP.device)
            output = model(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            target_pred[len(target_pred):len(target) - 1] = pred.tolist()
            target_real[len(target_real):len(target) - 1] = target.tolist()
        target_pred = np.array(target_pred)
        target_real = np.array(target_real)
    C = confusion_matrix(target_real, target_pred)
    plt.matshow(C, cmap=plt.cm.Blues)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(HP.matrix_path, dpi=600)

'''roc_auc_2class'''
def plot_roc_auc(model, test_dataloader, HP):
    model.eval()
    model = model.to(HP.device)
    target_real = []
    target_pred_prob = []

    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(HP.device)
            target = target.to(HP.device)
            output = model(data)
            probabilities = torch.softmax(output[1], dim=1)  # »ñÈ¡Ô¤²âÀà±ðµÄ¸ÅÂÊ
            target_pred_prob.extend(probabilities[:, 1].cpu().numpy())  # ½ö»ñÈ¡ÕýÀà±ðµÄ¸ÅÂÊ
            target_real.extend(target.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(target_real, target_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(HP.roc_auc_path, dpi=600)

'''roc_auc_nclass'''
def plot_roc_auc_multiclass(model, test_dataloader, HP):
    model.eval()
    model = model.to(HP.device)
    target_real = []
    target_pred_prob = []

    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(HP.device)
            target = target.to(HP.device)
            output = model(data)
            probabilities = torch.softmax(output[1], dim=1)
            target_pred_prob.extend(probabilities.cpu().numpy())
            target_real.extend(target.cpu().numpy())

    target_real = label_binarize(target_real, classes=list(range(HP.classes)))
    target_pred_prob = np.array(target_pred_prob)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(HP.classes):
        fpr[i], tpr[i], _ = roc_curve(target_real[:, i], target_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(target_real.ravel(), target_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=2, label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

def plot_roc_auc_multiclass_total(models, test_dataloaders, labels,colors,linewidths, HP):
    plt.figure(figsize=(5, 4.2))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})
    for model, test_dataloader, label, color, linewidth in zip(models, test_dataloaders, labels, colors, linewidths):
        model = torch.load(model)
        model.eval()
        model = model.to(HP.device)
        target_real = []
        target_pred_prob = []

        with torch.no_grad():
            for data, target in test_dataloader:
                target = target.long()
                data = data.to(HP.device)
                target = target.to(HP.device)
                output = model(data)
                probabilities = torch.softmax(output[1], dim=1)  # »ñÈ¡Ô¤²âÀà±ðµÄ¸ÅÂÊ
                target_pred_prob.extend(probabilities.cpu().numpy())
                target_real.extend(target.cpu().numpy())

        target_real = label_binarize(target_real, classes=np.unique(target_real))  # ½«¶àÀà±ð±êÇ©¶þÖµ»¯
        target_pred_prob = np.array(target_pred_prob)  # ½«ÁÐ±í×ª»»Îª NumPy Êý×é

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(target_real.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(target_real[:, i], target_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # ¼ÆËãmicroÆ½¾ùÖµ
        fpr["micro"], tpr["micro"], _ = roc_curve(target_real.ravel(), target_pred_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.plot(fpr["micro"], tpr["micro"], lw=linewidth, color=color, label=f'{label} (AUC = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.title('ROC Curve (r=%.2f)'%(HP.wrongratio))
    plt.legend(loc='lower right')
    plt.savefig(HP.roc_auc_path, dpi=1200)


'''画聚类特征可视化'''
def scatter(model, HP, test_dataloader):
    model.eval()
    model = model.to(HP.device)
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            target = target.squeeze()
            if torch.cuda.is_available():
                data = data.to(HP.device)
            output = model(data)
            feature_map[len(feature_map):len(output[0]) - 1] = output[0].tolist()
            target_output[len(target_output):len(target) - 1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
        tsne = TSNE(n_components=2)
        eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(feature_map))

        fig = plt.figure(figsize=(4, 3))
        plt.scatter(eval_tsne_embeds[:, 0], eval_tsne_embeds[:, 1], lw=0, s=20, c=target_output, cmap=plt.cm.get_cmap("jet", HP.classes))
        plt.colorbar(ticks=range(HP.classes))
        fig.savefig(HP.visualization_path, dpi=1200)


'''顺时针旋转矩阵'''
def rotate(matrix):
    matrix: List[List[int]]
    n = len(matrix)
    matrix_new = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix_new[j][n - i - 1] = matrix[i][j]
    matrix[:] = matrix_new
    return matrix

'''画混淆矩阵'''
def confusion_matrix_():
    data_pred = mat73.loadmat("pred_labels.mat")
    target_pred = data_pred['target_pred'][:]
    target_pred = target_pred.astype(np.uint8)
    data_real = mat73.loadmat("real_labels.mat")
    target_real = data_real['target_real'][:]
    target_real = target_real.astype(np.uint8)
    target_pred = np.array(target_pred)
    target_real = np.array(target_real)
    # 绘制
    C = confusion_matrix(target_real, target_pred)
    plt.matshow(C, cmap=plt.cm.Blues)

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.savefig(f"matshow/try.png", dpi=600)

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

class Data_augment(nn.Module):
    def __init__(self, rotate=False, flip=False, rotate_and_flip=False, awgn=False, add_noise=False, slice=False):
        super(Data_augment, self).__init__()

        self.rotate = rotate
        self.rotate_angle = [0,90,180,270]

        self.flip = flip

        self.rotate_and_flip = rotate_and_flip

        self.awgn = awgn
        self.noise_snr = [10, 20]  # 10~20

        self.add_noise = add_noise
        self.mean = 0
        self.std = 0.1

        self.slice = slice
        self.slice_len = 2400

    def rotation_2d(self, x, ang):
        x_aug = torch.zeros(x.shape)
        if ang == 0:
            x_aug = x
        elif ang == 90:
            x_aug[0, :] = -x[1, :]
            x_aug[1, :] = x[0, :]
        elif ang == 180:
            x_aug = -x
        elif ang == 270:
            x_aug[0, :] = x[1, :]
            x_aug[1, :] = -x[0, :]
        else:
            print("Wrong input for rotation!")
        return x_aug

    def get_normalized_vector(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def Rotate(self, x):
        x_rotate = torch.zeros(x.shape)
        for i in range(x.shape[0]):
            ang = self.rotate_angle[torch.randint(len(self.rotate_angle), [1])]
            x_rotate[i, :, :] = self.rotation_2d(x[i, :, :], ang)
        return x_rotate.cuda()

    def Flip(self, x):
        mul = [-1, 1]
        for i in range(x.shape[0]):
            I_mul = mul[torch.randint(len(mul), [1])]
            Q_mul = mul[torch.randint(len(mul), [1])]
            x[i, 0, :] = I_mul * x[i, 0, :]
            x[i, 1, :] = Q_mul * x[i, 1, :]
        return x

    def Rotate_and_Flip(self, x):
        choice_list = ['Rotate', 'Flip', 'Flip']  # Rotate的0和180与Flip中重复，所以Flip有4种，Rotate有2种，为了保持每种概率都是1/6，给Rotate1/3概率，给Flip2/3概率
        rotate_angle = [90, 270]
        mul = [-1, 1]
        for i in range(x.shape[0]):
            choice = choice_list[torch.randint(len(choice_list), [1])]
            if choice == 'Rotate':
                ang = rotate_angle[torch.randint(len(rotate_angle), [1])]
                x[i, :, :] = self.rotation_2d(x[i, :, :], ang)
            else:
                I_mul = mul[torch.randint(len(mul), [1])]
                Q_mul = mul[torch.randint(len(mul), [1])]
                x[i, 0, :] = I_mul * x[i, 0, :]
                x[i, 1, :] = Q_mul * x[i, 1, :]
        return x


    def Awgn(self, x, snr):
        '''
        加入高斯白噪声
        param snr；信噪比范围(db)
        param x: 原始信号
        param snr: 信噪比
        return: 加入噪声后的信号
        '''
        # np.random.seed(seed)  # 设置随机种子
        snr = torch.randint(snr[0], snr[1], [1])
        snr = 10 ** (snr / 10.0)
        x_awgn = torch.zeros((x.shape[0], x.shape[1]))
        # real
        x_real = x[:, 0]
        xpower_real = torch.sum(x_real ** 2) / len(x_real)
        npower_real = xpower_real / snr
        noise_real = torch.randn(len(x_real)) * torch.sqrt(npower_real)
        x_awgn[:, 0] = x_real + noise_real
        # imag
        x_imag = x[:, 1]
        xpower_imag = torch.sum(x_imag ** 2) / len(x_imag)
        npower_imag = xpower_imag / snr
        noise_imag = torch.randn(len(x_imag)) * torch.sqrt(npower_imag)
        x_awgn[:, 1] = x_imag + noise_imag
        return x_awgn

    def Add_noise(self, x):
        d = torch.normal(mean=self.mean, std=self.std, size=(x.shape[0], x.shape[1], x.shape[2]))
        return x + d

    def Slice(self, x):
        start = torch.randint(0, x.shape[2] - self.slice_len, [1])
        end = start + self.slice_len
        return x[:, :, start:end]

    def forward(self, x):
        x = x.cuda()
        if self.rotate:
            x = self.Rotate(x)
        if self.flip:
            x = self.Flip(x)
        if self.rotate_and_flip:
            x = self.Rotate_and_Flip(x)
        if self.awgn:
            x = self.Awgn(x)
        if self.add_noise:
            x = self.Add_noise(x)
        if self.slice:
            x = self.Slice(x)

        return x


def find_max_indices(arr):
    max_indices = [np.argmax(row).tolist() for row in arr]
    return max_indices


def calculate_confidence_threshold(y_true, y_pred_probs, num_classes):
    confidence_thresholds = []
    for j in range(num_classes):
        y_true_j = y_true == j
        y_pred_probs_max = np.array(find_max_indices(y_pred_probs)) == j

        k = len(y_true[y_true_j | y_pred_probs_max])
        #k = len(y_true[y_true_j & y_pred_probs_max])

        confidence_j = y_pred_probs[(y_true_j) & (y_pred_probs_max)][:, j].sum() / k

        confidence_thresholds.append(confidence_j)
    return confidence_thresholds


def filter_incorrect_samples(y_true, y_pred_probs, confidence_thresholds):
    incorrect_samples = []
    for i in range(len(y_true)):
        predicted_class = torch.argmax(y_pred_probs[i])
        true_class = y_true[i].item()

        if (predicted_class != true_class) or (predicted_class == true_class and y_pred_probs[i][true_class] <= confidence_thresholds[true_class]):
            incorrect_samples.append((i, true_class, predicted_class, y_pred_probs[i][true_class]))

    incorrect_samples.sort(key=lambda x: x[3], reverse=False)
    incorrect_index = [sample[0] for sample in incorrect_samples]

    return incorrect_index
