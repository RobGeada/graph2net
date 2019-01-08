import torch.nn as nn
from graph2net.helpers import *
from graph2net.ops import Zero


class Edge(nn.Module):
    def __init__(self, target, origin, op, send_frac):
        super().__init__()

        # edge attributes
        self.target = target
        self.target_name = target.name
        self.op = op

        # link target backwards
        target.previous_names.add(origin.name)

        # forward functionality
        self.op_function = None
        self.zero = Zero

        # bookkeeping
        self.node_out = None
        self.out_dim = None
        self.send_frac = send_frac
        self.out = None

    def forward(self, x, zero=False):
        return self.op_function(x) if not zero else self.zero(x)

    def get_num_params(self):
        return general_num_params(self)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        send_frac = "{:2.0f}%".format(self.send_frac * 100)
        return "{:<18} ({:>4})- > {:<50} -> {:<18} to Node {} ({:,} params)".format(str(self.node_out),
                                                                                    send_frac,
                                                                                    self.op,
                                                                                    str(self.out_dim),
                                                                                    self.target.name,
                                                                                    self.get_num_params())
