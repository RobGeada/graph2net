from graph2net.ops import *
from graph2net.helpers import *

powers = [(2 ** x) * 3 for x in range(3, 20)]


class Node(nn.Module):
    def __init__(self, name, dim=None):
        super(Node, self).__init__()

        # node attributes
        self.name = name
        self.dim = dim

        # node connections
        self.cnx = []
        self.ins = []
        self.out = None

    def add_cnx(self, cnx, o_pos, t_pos, cell_type):
        # ===DIMENSION MODIFICATION======================================
        in_channels = self.dim[1]

        # double channels if node receives cell input
        if o_pos == 0 and cell_type == 1:
            post_pad_channels = [x for x in powers if x > in_channels][0]
            cnx.padder = padder(in_channels, post_pad_channels)
        else:
            post_pad_channels = in_channels

        # halve spatial dimensions if node sends to output
        stride = 2 if (t_pos == 1 and cell_type == 1) else 1

        # correct target channels if operation is channel modification
        if 'channel' in cnx.op:
            op_scaler = 2 if 'double' in cnx.op else .5
        else:
            op_scaler = 1
        post_op_channels = int(post_pad_channels * op_scaler)

        # ===Intercept channel modifications if they are incompatible with target node
        if cnx.target.dim is not None:
            tgt_channels = cnx.target.dim[1]

            if tgt_channels != post_op_channels:
                cnx.padder = padder(in_channels, int(tgt_channels / op_scaler))
                post_pad_channels = int(tgt_channels / op_scaler)
                post_op_channels = tgt_channels

        # ===CONNECTION SETTING==========================================
        # set attributes for connection
        cnx.op_function = commons[cnx.op](post_pad_channels, stride=stride)
        cnx.in_dim = self.dim
        cnx.out_dim = [self.dim[0],
                       post_op_channels,
                       int(self.dim[2] / stride),
                       int(self.dim[3] / stride)]
        if in_channels == post_pad_channels:
            cnx.op = "{:<15} (by {}, {}----->{}-op->{})".format(cnx.op, stride, in_channels, post_pad_channels,
                                                                post_op_channels)
        else:
            cnx.op = "{:<15} (by {}, {}-pad->{}-op->{})".format(cnx.op, stride, in_channels, post_pad_channels,
                                                                post_op_channels)

        # ===SANITY CHECKING=============================================
        # make sure tensors are of right dimensionality
        if cnx.target.dim == None or cnx.target.dim == cnx.out_dim:
            cnx.target.dim = cnx.out_dim
        else:
            raise ValueError("Dimension mismatch from {} to {}: Target Dim {} vs Edge Dim {}".format(
                self.name,
                cnx.target.name,
                cnx.target.dim,
                cnx.out_dim))

        self.cnx.append(cnx)

    def forward(self, x=None, verbose=False):
        # if cell connects to first node, pass in input
        if not self.ins:
            self.ins = [x]

        if verbose: print("=== Node ", self.name, "===")
        if verbose: print("Node Inputs:", [x.shape for x in self.ins])

        # add all input tensors together
        self.out = sum(self.ins)

        # clear memory of inputs (to avoid "retain-graph=True error")
        self.ins = []

        # pass data through subsequent connections and set node output
        for cnx in self.cnx:
            if verbose: print("{}->{} via {}: {}->".format(self.name, cnx.target.name, cnx.op, self.out.shape), end="")
            cnx.out = cnx(self.out)
            cnx.target.ins.append(cnx.out)
            if verbose: print("{}".format(cnx.out.shape))
        return self.out

    def __repr__(self):
        out = ""
        out += "{} Node {} {}\n".format(eq_string(4), self.name, eq_string(4))
        if len(self.cnx) == 0:
            out += "{:<18} -> out\n".format(str(self.dim))
        else:
            for cnx in self.cnx:
                out += str(cnx) + "\n"
        return out
