import torch
import torch.nn as nn

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

        if self.batch_average:
            loss /= n

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

        if self.batch_average:
            loss /= n

        return loss

class DomainLosses(object):
    def __init__(self, batch_average=True, cuda=False):
        self.batch_average = batch_average
        self.cuda = cuda
    def build_loss(self, mode='no_inv'):
        """Choices: ['no_inv' or 'inv']"""
        if mode == 'no_inv':
            return self.DomainClassiferLoss
        elif mode == 'inv':
            return self.DomainInvLoss
        else:
            raise NotImplementedError

    def DomainClassiferLoss(self, src_logit, tgt_logit):
        assert src_logit.size() == tgt_logit.size()
        n1, c1, h1, w1 = src_logit.size()
        n2, c2, h2, w2 = tgt_logit.size()
        src_target = torch.ones([n1,h1,w1], dtype=src_logit.dtype, layout=src_logit.layout, device=src_logit.device)
        tgt_target = torch.zeros([n2,h2,w2], dtype=tgt_logit.dtype, layout=tgt_logit.layout, device=tgt_logit.device)
        logit = torch.cat([src_logit,tgt_logit],dim=0)
        target = torch.cat([src_target,tgt_target],dim=0)
        criterion = nn.CrossEntropyLoss(reduction = 'mean')
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= (n1 + n2)

        return loss

    def DomainInvLoss(self, src_logit, tgt_logit):
        assert src_logit.size() == tgt_logit.size()
        n1, c1, h1, w1 = src_logit.size()
        n2, c2, h2, w2 = tgt_logit.size()
        src_target = torch.zeros([n1,h1,w1], dtype=src_logit.dtype, layout=src_logit.layout, device=src_logit.device)
        tgt_target = torch.ones([n2,h2,w2], dtype=tgt_logit.dtype, layout=tgt_logit.layout, device=tgt_logit.device)
        logit = torch.cat([src_logit,tgt_logit],dim=0)
        target = torch.cat([src_target,tgt_target],dim=0)
        criterion = nn.CrossEntropyLoss(reduction = 'mean')
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= (n1 + n2)

        return loss











# if __name__ == "__main__":
#     loss = SegmentationLosses(cuda=True)
#     a = torch.rand(1, 3, 7, 7).cuda()
#     b = torch.rand(1, 7, 7).cuda()
#     print(loss.CrossEntropyLoss(a, b).item())
#     print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
#     print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())





