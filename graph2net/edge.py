import torch
import torch.nn as nn

from graph2net.helpers import *


class Edge(nn.Module):
    def __init__(self, target, op):
        super(Edge, self).__init__()

        # edge attributes
        self.target = target
        self.op = op

        # forward functionality
        self.op_function = None
        self.padder = lambda x: x

        # bookkeeping
        self.in_dim = None
        self.out_dim = None
        self.out = None

    def forward(self, x):
        x = self.padder(x)
        return self.op_function(x)

    def get_num_params(self):
        return general_num_params(self)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{:<18} -> {:<30} -> {:<18} to Node {} ({:,} params)".format(str(self.in_dim),
                                                                            self.op,
                                                                            str(self.out_dim),
                                                                            self.target.name,
                                                                            self.get_num_params())
