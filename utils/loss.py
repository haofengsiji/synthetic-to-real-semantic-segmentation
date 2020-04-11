import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction = 'mean')
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())


        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction = 'mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt


        return loss

class DomainLosses(object):
    def __init__(self, batch_average=True, cuda=False):
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self):

        return self.DomainClassiferLoss

    def DomainClassiferLoss(self, source, target):
        assert source.size() == target.size()
        source_p_theta = F.softmax(source, dim=1)
        target_p_theta = F.softmax(target, dim=1)
        loss = torch.mean(-torch.log(source_p_theta[0]) - torch.log(1 - target_p_theta[0]))
        acc = torch.mean(((source>=0.5).float() + (target<0.5).float())/2)

        return loss,acc

if __name__ == "__main__":
    loss = DomainLosses(cuda=True)
    a = torch.ones(1, 1, 7, 7).cuda()*0.1
    b = torch.ones(1, 1, 7, 7).cuda()*0.9
    d_loss,acc = loss.DomainClassiferLoss(b, a)
    print(d_loss,acc)





