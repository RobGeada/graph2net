from graph2net.ops import *
from graph2net.helpers import *
import torch
import gc


class Node(nn.Module):
    def __init__(self, name, cell_type, concatenate, input_node=False, output_node=False, dim=None):
        super().__init__()

        # node attributes
        self.name = name
        self.is_input_node = input_node
        self.is_output_node = output_node
        self.cell_type = cell_type

        # node internal operations
        self.concatenate = concatenate
        self.padder = None

        # connection dimensionality
        self.dim = [dim] if dim is not None else []
        self.in_channels = None
        self.post_pad_channels = None

        # node inputs
        self.previous_names = set([])
        self.ins = []

        # node outputs
        self.cnx = []
        self.send_fraction = {}
        self.chunk_sizes = []
        self.target_names_set = set([])
        self.target_names_list = []
        self.num_targets = 0
        self.out = None

    def set_post_pad_channels(self):
        self.in_channels =  sum([x[1] for x in self.dim]) if self.concatenate else self.dim[0][1]

        # double channels if node receives cell input
        if self.is_input_node and self.cell_type == 1:
            self.post_pad_channels = self.in_channels * 2
        else:
            self.post_pad_channels = self.in_channels
        self.padder = padder(self.in_channels, self.post_pad_channels)

    def add_cnx(self, cnx, send_frac):
        cnx_in_channels = int(self.post_pad_channels * send_frac)
        self.chunk_sizes += [cnx_in_channels]

        # correct target channels if operation is channel modification and target channels are valid
        if 'channel' in cnx.op:
            op_scaler = 2 if 'double' in cnx.op else .5
            is_summation = not cnx.target.concatenate
            target_dim_set = not not cnx.target.dim

            if target_dim_set and is_summation:
                if cnx.target.dim[0][1] != int(self.post_pad_channels * op_scaler):
                    op_scaler = 1
        else:
            op_scaler = 1
        post_op_channels = int(cnx_in_channels * op_scaler)

        # add operation along connection edge
        stride = 2 if (cnx.target.is_output_node and self.cell_type == 1) else 1
        cnx.op_function = commons[cnx.op](cnx_in_channels, stride=stride)
        cnx.node_out = self.dim[0]
        cnx.out_dim = width_mod(channel_mod(self.dim[0], post_op_channels), stride)

        pad_str = "----->{}".format(self.post_pad_channels) if self.in_channels == self.post_pad_channels \
            else "-pad->{}".format(self.post_pad_channels)
        split_str = "--->{}".format(cnx_in_channels) if self.post_pad_channels == cnx_in_channels \
            else "-/->{}".format(cnx_in_channels)

        cnx.op = "{:<15} (by {}, {}{}{}-op->{})".format(cnx.op,
                                                        stride,
                                                        self.in_channels,
                                                        pad_str,
                                                        split_str,
                                                        post_op_channels)

        # add connection into target dimensionality
        if not cnx.target.dim:
            cnx.target.dim = [cnx.out_dim]
        else:
            cnx.target.dim += [cnx.out_dim]

        self.cnx.append(cnx)
        self.target_names_set.add(cnx.target.name)
        self.target_names_list.append(cnx.target.name)
        self.num_targets += 1

    def forward(self, x=None, node_cnx={}, verbose=False):
        # if cell connects to first node, pass in input
        if not self.ins:
            self.ins = [x]

        # add all input tensors together
        if self.concatenate:
            self.out = torch.cat(self.ins, 1)
        else:
            self.out = sum(self.ins)

        # clear memory of inputs (to avoid "retain-graph=True error")
        self.ins = []

        # pass data through padder
        self.out = self.padder(self.out)
        chunks = [] if self.chunk_sizes == [] else torch.split(self.out, min(self.chunk_sizes), 1)
        chunk_iterator = 0

        # pass data through subsequent connections and set node output
        if not node_cnx:
            for cnx in self.cnx:
                if self.send_fraction[cnx.target_name] is 1:
                    cnx.out = cnx(self.out)
                else:
                    cnx.out = cnx(chunks[chunk_iterator])
                    chunk_iterator += 1
                cnx.target.ins.append(cnx.out)
        else:
            name = self.name
            for cnx in self.cnx:
                if node_cnx.get(name, cnx.target_name):
                    if self.send_fraction[cnx.target_name] is 1:
                        cnx.out = cnx(self.out)
                    else:
                        cnx.out = cnx(chunks[chunk_iterator])
                        chunk_iterator += 1
                else:
                    cnx.out = cnx(self.out, zero=True)
                cnx.target.ins.append(cnx.out)
        return self.out

    def __repr__(self):
        out = ""
        node_type = "concat" if self.concatenate else "sum"
        out += "{} Node {} ({}) {}\n".format(eq_string(4), self.name, node_type, eq_string(4))
        if len(self.cnx) == 0:
            out += "{:<18} -> out\n".format(str(self.dim))
        else:
            for cnx in self.cnx:
                out += str(cnx) + "\n"
        return out
