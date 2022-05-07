'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 模型需继承nn.Module
class VGG(nn.Module):
    # 初始化参数：
    def __init__(self, vgg_name, num_classes, batch_size=64):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(4096, num_classes)
        self.batch_size = batch_size

    # 模型计算时的前向过程，也就是按照这个过程进行计算
    def forward(self, x):
        x = x.view(self.batch_size, 1, -1)
        # print('x', x.shape)
        out = self.features(x)
        # print('out', out.shape)
        out = out.view(self.batch_size, -1)
        # out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm1d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool1d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
if __name__ == '__main__':
    with torch.no_grad():
        num_classes = 11
        net = VGG('VGG16', num_classes=num_classes, batch_size=110).cuda()
        print(net)
        x = torch.randn(110, 1, 128, 2).cuda()
        y = net(x)
        print('y', y.shape)
