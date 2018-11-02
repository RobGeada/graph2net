import torch.nn as nn
from graph2net.helpers import *
from graph2net.ops import Zero


class Edge(nn.Module):
    def __init__(self, target, origin, op):
        super(Edge, self).__init__()

        # edge attributes
        self.target = target
        self.target_name = target.name
        self.op = op

        # link target backwards
        target.previous_names.add(origin.name)

        # forward functionality
        self.op_function = None
        self.padder = lambda x: x
        self.zero = Zero

        # bookkeeping
        self.in_dim = None
        self.out_dim = None
        self.out = None

    def forward(self, x, zero=False):
        if not zero:
            x = self.padder(x)
            return self.op_function(x)
        else:
            return self.zero(x)

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
