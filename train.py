from model_complexcnn_sparse import sparse_complex_model
from optimizer_SGD_APGD import APGD, SGD
from utils import *
from config import *
from model_complexcnn_onlycnn import base_complex_model
from dataload_ADSB import wrongdataset
from loss import LabelSmoothingLoss, Symmetric_Cross_Entropy, Generalized_Cross_Entropy, MAELoss, CenterLoss

def CrossEntropyLoss(HP):
   # load model
   model = base_complex_model()
   model = model.to(HP.device)
   # set loss
   loss = nn.CrossEntropyLoss(reduction='sum')
   loss = loss.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=0)
   # train
   #train_and_val(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,optimizer=optim, HP=HP)
   # test
   model = torch.load(HP.save_path)
   evaluate(model, test_dataloader, HP=HP)
   confusion_matrix_mat(model, test_dataloader, HP=HP)
   #plot_roc_auc_multiclass(model, test_dataloader, HP=HP)
   scatter(model, HP, test_dataloader)

def Mixup(HP):
   # load model
   model = base_complex_model()
   model = model.to(HP.device)
   # set loss
   loss = nn.CrossEntropyLoss(reduction='sum')
   loss = loss.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=0)
   # train
   mixup_train_and_val(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,optimizer=optim, HP=HP)
   # test
   model = torch.load(HP.save_path)
   evaluate(model, test_dataloader, HP=HP)


def LabelSmoothingLoss_train(HP):
   # load model
   model = base_complex_model()
   model = model.to(HP.device)
   # set loss
   loss = LabelSmoothingLoss(Dataset.classes, smoothing=HP.smoothing, dim=-1)
   loss = loss.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=0)
   # train
   #train_and_val(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,optimizer=optim, HP=HP)
   # test
   model = torch.load(HP.save_path)
   evaluate(model, test_dataloader, HP=HP)
   scatter(model, HP, test_dataloader)

def Generalized_Cross_Entropy_train(HP):
   # load model
   model = base_complex_model()
   model = model.to(HP.device)
   # set loss
   loss = Generalized_Cross_Entropy(Dataset.classes, HP.r)
   loss = loss.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=0)
   # train
   train_and_val(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,optimizer=optim, HP=HP)#    train_and_val(model, SCE_train, SCE_val, train_dataloader, val_dataloader)
   # test
   model = torch.load(HP.save_path)
   evaluate(model, test_dataloader, HP=HP)

def Symmetric_Cross_Entropy_train(HP):
   # load model
   model = base_complex_model()
   model = model.to(HP.device)
   # set loss
   loss = Symmetric_Cross_Entropy(HP.alpha, HP.beta, Dataset.classes)
   loss = loss.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=0)
   # train
   train_and_val(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,optimizer=optim, HP=HP)#    train_and_val(model, SCE_train, SCE_val, train_dataloader, val_dataloader)
   # test
   model = torch.load(HP.save_path)
   evaluate(model, test_dataloader, HP=HP)


def Mean_Absolute_Error(HP):
   # load model
   model = base_complex_model()
   model = model.to(HP.device)
   # set loss
   loss = MAELoss(Dataset.classes, HP.scale)
   loss = loss.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=0)
   # train
   train_and_val(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optim, HP=HP)  # train_and_val(model, SCE_train, SCE_val, train_dataloader, val_dataloader)
   # test
   model = torch.load(HP.save_path)
   evaluate(model, test_dataloader, HP=HP)


def metirc_learning(HP):
   # load model
   model = base_complex_model()
   model = model.to(HP.device)
   # set loss
   loss1 = nn.CrossEntropyLoss(reduction='sum')
   loss2 = CenterLoss(HP.num_classes,HP.feat_dim,HP.device)
   loss1 = loss1.to(HP.device)
   loss2 = loss2.to(HP.device)
   # set optimizer
   optim1 = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=0)
   optim2 = torch.optim.Adam(loss2.parameters(), lr=HP.lr, weight_decay=0)
   # train
   metric_learning_train_and_val(model, loss_function1=loss1, loss_function2=loss2,train_dataloader=train_dataloader, val_dataloader=val_dataloader,optimizer1=optim1,optimizer2=optim2, HP=HP)
   # test
   model = torch.load(HP.save_path)
   evaluate(model, test_dataloader, HP=HP)


def cleanlab_Semi_Supervised_otherloss(HP):
   # load model
   model1 = base_complex_model()
   model1 = model1.to(HP.device)
   # set loss
   if HP.loss_type == "CE":
      loss = nn.CrossEntropyLoss(reduction='sum')
   if HP.loss_type == "GCE":
      loss = Generalized_Cross_Entropy(Dataset.classes, HP.r)
   if HP.loss_type == "SCE":
      loss = Symmetric_Cross_Entropy(HP.alpha, HP.beta, Dataset.classes)
   loss0 = loss
   loss0 = loss0.to(HP.device)
   loss1 = loss
   loss1 = loss1.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model1.parameters(), lr=HP.lr, weight_decay=0)
   # train
   train_dataloader_new, val_dataloader_new,= cleanlab_Semi_Supervised_train_and_val(model1, loss_function0=loss0,loss_function1=loss1, train_dataloader_orignal=all_train_dataloader, val_dataloader_orignal=val_dataloader, optimizer=optim, HP=HP)
   # test
   model2 = base_complex_model().to(HP.device)
   loss2 = loss
   loss2 = loss2.to(HP.device)
   optim = torch.optim.Adam(model2.parameters(), lr=HP.lr, weight_decay=0)
   train_and_val(model2, loss_function=loss2, train_dataloader=train_dataloader_new, val_dataloader=val_dataloader_new,optimizer=optim, HP=HP)
   model2 = torch.load(HP.save_path)
   evaluate(model2, test_dataloader, HP=HP)

def cleanlab_Semi_Supervised_LSR(HP):
   # load model
   model1 = base_complex_model()
   model1 = model1.to(HP.device)
   # set loss
   loss0 = LabelSmoothingLoss(Dataset.classes, smoothing=HP.smoothing, dim=-1)
   loss0 = loss0.to(HP.device)

   loss1 = LabelSmoothingLoss(Dataset.classes, smoothing=HP.smoothing, dim=-1)
   loss1 = loss1.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model1.parameters(), lr=HP.lr, weight_decay=0)
   # train
   train_dataloader_new, val_dataloader_new,= cleanlab_Semi_Supervised_train_and_val(model1, loss_function0=loss0,loss_function1=loss1, train_dataloader_orignal=all_train_dataloader, val_dataloader_orignal=val_dataloader, optimizer=optim, HP=HP)
   #test
   model2 = base_complex_model().to(HP.device)
   loss2 = LabelSmoothingLoss(Dataset.classes, smoothing=HP.smoothing, dim=-1)
   loss2 = loss2.to(HP.device)
   optim = torch.optim.Adam(model2.parameters(), lr=HP.lr, weight_decay=0)
   train_and_val(model2, loss_function=loss2, train_dataloader=train_dataloader_new, val_dataloader=val_dataloader_new,optimizer=optim, HP=HP)
   model2 = torch.load(HP.save_path)
   evaluate(model2, test_dataloader, HP=HP)
   confusion_matrix_mat(model2, test_dataloader, HP=HP)

def sparse_cleanlab_Semi_Supervised_LSR(HP):
   # load model
   model1 = base_complex_model()
   model1 = model1.to(HP.device)
   # set loss
   loss0 = LabelSmoothingLoss(Dataset.classes, smoothing=HP.smoothing, dim=-1)
   loss0 = loss0.to(HP.device)

   loss1 = LabelSmoothingLoss(Dataset.classes, smoothing=HP.smoothing, dim=-1)
   loss1 = loss1.to(HP.device)
   # set optimizer
   optim = torch.optim.Adam(model1.parameters(), lr=HP.lr, weight_decay=0)
   # train
   train_dataloader_new, val_dataloader_new, = cleanlab_Semi_Supervised_train_and_val(model1, loss_function0=loss0,
                                                                                      loss_function1=loss1,
                                                                                      train_dataloader_orignal=all_train_dataloader,
                                                                                      val_dataloader_orignal=val_dataloader,
                                                                                      optimizer=optim, HP=HP)
   # test
   model2 = sparse_complex_model(1024).to(HP.device)
   loss2 = LabelSmoothingLoss(Dataset.classes, smoothing=HP.smoothing, dim=-1)
   loss2 = loss2.to(HP.device)
   # optimizer1 = SGD([{'params': model2.parameters()}], lr=HP.lr, momentum=0.9, weight_decay=0.0001)
   # optimizer2 = APGD([{'params': model2.lamda}], alpha=HP.alpha, device=HP.device, lr=HP.lr, momentum=0.9,weight_decay=0.0001)
   optimizer1 = SGD([{'params': [param for name, param in model2.named_parameters() if 'lamda' not in name]}], lr=HP.lr, momentum=0.9, weight_decay=0.0001)
   optimizer2 = APGD([{'params': [param for name, param in model2.named_parameters() if 'lamda' in name]}], alpha=HP.alpha, device=HP.device, lr=HP.lr, momentum=0.9,weight_decay=0.0001)
   sparse_train_and_val(model2, loss_function=loss2, train_dataloader=train_dataloader_new, val_dataloader=val_dataloader_new, optimizer1=optimizer1, optimizer2=optimizer2, HP=HP)
   model_prune = sparse_complex_model
   prune_model_and_test(model_prune, HP, test_dataloader)


if __name__ == '__main__':
   # 设置随机种子
   set_seed(ALL.seed)
   # load dataset
   all_train_dataloader, train_dataloader, val_dataloader, test_dataloader = Data_prepared(Dataset)

   #CrossEntropyLoss(CE)
   #Mixup(mixup)
   #LabelSmoothingLoss_train(LSR)
   #Generalized_Cross_Entropy_train(GCE)
   #Mean_Absolute_Error(MAE)
   #Symmetric_Cross_Entropy_train(SCE)
   #metirc_learning(metric)

   #our method--SSR-SEI
   cleanlab_Semi_Supervised_LSR(cleanlab_semi_supervised_LSR)
   #cleanlab_Semi_Supervised_LSR(ASS_onetime)
   #cleanlab_Semi_Supervised_otherloss(cleanlab_semi_supervised_otherloss)
   #sparse_cleanlab_Semi_Supervised_LSR(sparse_cleanlab_semi_supervised_LSR)

   #plot_roc_auc
   # models = [CE.save_path,mixup.save_path,LSR.save_path,GCE.save_path,SCE.save_path,metric.save_path,cleanlab_semi_supervised_LSR.save_path]
   # test_dataloaders = [test_dataloader,test_dataloader,test_dataloader,test_dataloader,test_dataloader,test_dataloader,test_dataloader]
   # labels = ['CE','Mixup','LSR','GCE','SCE','DML','SSR']
   # colors = ['blue','green','orange','purple','brown','pink','red']
   # linewidths = [1, 1, 1, 1, 1, 1, 2]
   # plot_roc_auc_multiclass_total(models, test_dataloaders, labels,colors,linewidths, Dataset)
