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
        n, c, h, w = source.size()
        real = torch.ones([n, h, w], dtype=source.dtype, layout=source.layout, device=source.device)
        log_source = -F.nll_loss(source_p_theta,real.long())
        log_target = -F.nll_loss(1 - target_p_theta.data,real.long())

        sum_source = log_source.sum()
        sum_target = log_target.sum()
        acc = torch.mean((torch.argmax(source,dim=1).float() + (1 - torch.argmax(target,dim=1).float()))/2).item()
        return sum_source + sum_target,acc

    # def DomainClassiferLoss(self, src_logit, tgt_logit):
    #     assert src_logit.size() == tgt_logit.size()
    #     n1, c1, h1, w1 = src_logit.size()
    #     n2, c2, h2, w2 = tgt_logit.size()
    #     src_target = torch.zeros([n1, h1, w1], dtype=src_logit.dtype, layout=src_logit.layout, device=src_logit.device)
    #     tgt_target = torch.ones([n2, h2, w2], dtype=tgt_logit.dtype, layout=tgt_logit.layout, device=tgt_logit.device)
    #     criterion = nn.CrossEntropyLoss(reduction='mean')
    #     if self.cuda:
    #         criterion = criterion.cuda()
    #     loss = criterion(src_logit, src_target.long()) + criterion(tgt_logit, tgt_target.long())
    #     acc = (torch.sum(1 - torch.argmax(src_logit,dim=1)) + torch.sum(torch.argmax(tgt_logit,dim=1))).float()/2/n1/h1/w1
    #
    #     return loss, acc


# if __name__ == "__main__":
#     loss = SegmentationLosses(cuda=True)
#     a = torch.rand(1, 3, 7, 7).cuda()
#     b = torch.rand(1, 7, 7).cuda()
#     print(loss.CrossEntropyLoss(a, b).item())
#     print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
#     print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

if __name__ == "__main__":
    loss = DomainLosses(cuda=True)
    a = torch.ones(1, 1, 7, 7).cuda()*0
    b = torch.ones(1, 1, 7, 7).cuda()*1
    src = torch.cat([a,b],dim=1)
    tgt = torch.cat([b,a],dim=1)
    d_loss,acc = loss.DomainClassiferLoss(src, tgt)
    print(d_loss,acc)





