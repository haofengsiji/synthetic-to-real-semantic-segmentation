import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class DomainClassifer(nn.Module):
    def __init__(self,backbone,BatchNorm,level='high'):
        super(DomainClassifer, self).__init__()
        if backbone == 'mobilenet' and level == 'high':
            in_channel = 256
        else:
            raise NotImplementedError

        self.DC_adnn1 = nn.Sequential(nn.Conv2d(in_channel, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(1024),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
        self.DC_adnn2 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                                      BatchNorm(1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))
        self.DC_adnn3 = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.Sigmoid())


        self._init_weight()
    def forward(self,input):
        output = self.DC_adnn1(input)
        output = self.DC_adnn2(output)
        output = self.DC_adnn3(output)

        return output


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_domaincls(backbone, BatchNorm):
    return DomainClassifer(backbone, BatchNorm)

if __name__ == '__main__':
    BN = SynchronizedBatchNorm2d
    model = DomainClassifer('mobilenet', BN)
    model.eval()
    input = torch.rand(1, 256, 32, 32)
    output = model(input)
    print(output.size())