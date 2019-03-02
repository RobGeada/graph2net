import torch.nn as nn
from graph2net.helpers import *
import ipdb

def bracket(ops, ops2=None):
    first_relu = False
    inplace = False
    affine = True


    if first_relu:
        out = [nn.ReLU(inplace=inplace)] + ops + [nn.BatchNorm2d(ops[-1].out_channels, affine=affine)]
        if ops2:
            out += [nn.ReLU(inplace=inplace)] + ops2 + [nn.BatchNorm2d(ops2[-1].out_channels, affine=affine)]
        return nn.Sequential(*out)
    else:
        out = ops + [nn.BatchNorm2d(ops[-1].out_channels, affine=affine)]
        if ops2:
            out += [nn.ReLU(inplace=inplace)] + ops2 + [nn.BatchNorm2d(ops2[-1].out_channels, affine=affine)]
        return nn.Sequential(*out)


class DilatedConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = bracket([
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False)
        ])

    def forward(self, x):
        return self.op(x)


class SingleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        if c_in == c_out:
            self.op = bracket([
                nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            ])
        else:
            self.op = bracket([
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            ])

    def forward(self, x):
        return self.op(x)


class RectangleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        self.op = bracket([
            nn.Conv2d(c_in, c_in, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=(kernel_size, 1), padding=(padding, 0), bias=False)
        ])

    def forward(self, x):
        return self.op(x)


class SeparableConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        self.op = bracket([
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ], [
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=1, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ])

    def forward(self, x):
        return self.op(x)


class MinimumIdentity(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.out_channels = c_out

        if c_in == c_out:
            if stride == 1:
                self.identity = nn.Sequential()
            else:
                self.identity = nn.MaxPool2d(1, stride=stride)
        else:
            self.identity = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.identity(x)


class Zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return  x.mul(0.)


class NNView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Classifier(nn.Module):
    def __init__(self, in_size, out_size, name='Classifier'):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.name = name

        # '''
        self.op = nn.Sequential(
            NNView(),
            nn.Linear(in_size, out_size)
        )
        '''
        self.op = nn.Sequential(
            nn_View(),
            nn.Dropout(),
            nn.Linear(in_size, in_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_size, out_size),
            nn.Softmax(dim=1)
        )'''
        # '''

    def forward(self, x):
        return self.op(x)

    def get_param_counts(self):
        return [general_num_params(op) for op in self.op]

    def __repr__(self):
        params = self.get_param_counts()
        out = "{}{:^15}{}\n".format(eq_string(50), self.name.upper(), eq_string(50))
        '''
        out += "nn_View:    {:<6} -> {:<6} ({:,} params)\n".format(self.in_size, self.in_size, params[0])
        out += "Dropout:    {:<6} -> {:<6} ({:,} params)\n".format(self.in_size, self.in_size, params[1])
        out += "Linear:     {:<6} -> {:<6} ({:,} params)\n".format(self.in_size, self.in_size, params[2])
        out += "ReLU:       {:<6} -> {:<6} ({:,} params)\n".format(self.in_size, self.in_size, params[3])
        out += "Linear:     {:<6} -> {:<6} ({:,} params)\n".format(self.in_size, self.out_size, params[4])
        out += "Softmax:    {:<6} -> {:<6} ({:,} params)\n".format(self.out_size, self.out_size, params[5])
        '''
        out += "{}:  {:<6} -> {:<6} ({:,} params)".format(self.name, self.in_size, self.out_size, sum(params))
        return out


def padder(c_in, c_out, stride=1):
    return MinimumIdentity(c_in, c_out, stride=stride)


def initializer(c_in, c_out):
    return SingleConv(c_in, c_out, kernel_size=3, stride=1, padding=padsize(s=1))


def normalizer(c_in):
    return nn.BatchNorm2d(c_in, affine=False)


def padsize(s, k=3, d=1):
    pad = math.ceil((k * d - d + 1 - s) / 2)
    return pad


commons = {
    'zero': Zero(),
    'identity': lambda c_in, stride: MinimumIdentity(c_in, c_in, stride=stride),
    'double_channel': lambda c_in, stride: MinimumIdentity(c_in, c_in * 2, stride=stride),
    'halve_channel': lambda c_in, stride: MinimumIdentity(c_in, int(c_in * .5), stride=stride),
    'avg_pool_3x3': lambda c_in, stride: nn.AvgPool2d(3, stride=stride, padding=padsize(s=stride)),
    'max_pool_3x3': lambda c_in, stride: nn.MaxPool2d(3, stride=stride, padding=padsize(s=stride)),
    'max_pool_5x5': lambda c_in, stride: nn.MaxPool2d(5, stride=stride, padding=padsize(s=stride, k=5)),
    'max_pool_7x7': lambda c_in, stride: nn.MaxPool2d(7, stride=stride, padding=padsize(s=stride, k=7)),
    '1x7_7x1_conv': lambda c_in, stride: RectangleConv(c_in, c_in, 7, stride=stride, padding=padsize(s=stride, k=7)),
    '1x3_3x1_conv': lambda c_in, stride: RectangleConv(c_in, c_in, 3, stride=stride, padding=padsize(s=stride)),
    'dil_conv_3x3': lambda c_in, stride: DilatedConv(c_in, c_in, 3, stride=stride, padding=padsize(s=stride, d=2),
                                                     dilation=2),
    'conv_1x1': lambda c_in, stride: SingleConv(c_in, c_in, 1, stride=stride, padding=padsize(s=stride, k=1)),
    'conv_3x3': lambda c_in, stride: SingleConv(c_in, c_in, 3, stride=stride, padding=padsize(s=stride)),
    'sep_conv_3x3': lambda c_in, stride: SeparableConv(c_in, c_in, 3, stride=stride, padding=padsize(s=stride)),
    'sep_conv_5x5': lambda c_in, stride: SeparableConv(c_in, c_in, 5, stride=stride, padding=padsize(s=stride, k=5)),
    'sep_conv_7x7': lambda c_in, stride: SeparableConv(c_in, c_in, 7, stride=stride, padding=padsize(s=stride, k=7)),
}

idx_to_op = dict(enumerate(list(commons.keys())))
op_to_idx = {op: idx for idx, op in idx_to_op.items()}


def print_operations():
    for idx, op in idx_to_op.items():
        print(idx, ":", op)
