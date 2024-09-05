import torch
import torch.nn.functional as F

'''LabelSmoothingLoss'''
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        # 置信度为 1.0 减去平滑系数，用于计算targrt的概率
        self.confidence = 1.0 - smoothing
        # 平滑系数，用于计算pred
        self.smoothing = smoothing
        # 类别数
        self.cls = num_classes
        # 进行 softmax 计算的维度
        self.dim = dim

    def forward(self, pred, target):
        # 对预测值进行 log_softmax 操作，使得概率更加平滑
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # 创建一个和pred相同的全零张量 true_dist
            true_dist = torch.zeros_like(pred)
            # 对于每个样本的真实标签，将平滑系数分配到该标签对应的位置上
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # 将置信度分配到真实标签对应的位置上
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 计算交叉熵损失，true_dist 为平滑后的真实标签分布
        return torch.sum(-true_dist * pred, dim=self.dim).sum()

'''Generalized_Cross_Entropy'''
class Generalized_Cross_Entropy(torch.nn.Module):
    def __init__(self, num_classes,r):
        super(Generalized_Cross_Entropy, self).__init__()
        self.r = r
        self.num_classes = num_classes
    def forward(self, pred, target):
        # 计算p和y的alpha次方
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min =1e-8, max=1.0)
        label_one_hot = F.one_hot(target, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.r)) / self.r
        return loss.sum()

'''Symmetric_Cross_Entropy'''
class Symmetric_Cross_Entropy(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(Symmetric_Cross_Entropy, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='sum')
    def forward(self, pred, target):
        # CE
        ce = self.cross_entropy(pred, target)
        # RCE
        pred = F.softmax(pred, dim=1)
        #pred = torch.clamp(pred, min=1e-10, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).sum()
        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss

'''Mean Absolute Error'''
class MAELoss(torch.nn.Module):
    def __init__(self, num_classes, scale=2.0):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(target, self.num_classes).float()
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * loss.sum()


class CenterLoss(torch.nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        '''
        torch.nn.Parameter可以将一个不可训练的类型Tensor转换成可以训练的类型parameter，并将这个parameter绑定到module里面
        '''
        self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
        #self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='sum')

    def forward(self, pred, target):
        """
        Args:
            pred: feature matrix with shape (batch_size, feat_dim).
            target: ground truth labels with shape (batch_size).
        """
        # CE
        #ce_loss = self.cross_entropy(pred, target)
        '''
        torch.pow(x,y)：实现张量和标量之间逐元素求指数操作
        .t()：转置
        distmat.addmm_(1, -2, x, self.centers.t())  --->  1*distmat+(-2) * x * centers.t()
        '''
        batch_size = pred.size(0)
        distmat = torch.pow(pred, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, pred, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = target.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        center_loss = dist.clamp(min=1e-12, max=1e+12).sum()
        return center_loss