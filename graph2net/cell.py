from graph2net.node import Node
from graph2net.ops import *
from graph2net.edge import Edge
from graph2net.helpers import *

import random


class Cell(nn.Module):
    def __init__(self, layer, in_dim, cell, cell_type=0, verbose=False):
        super().__init__()
        self.in_dim = in_dim
        self.layer = layer
        self.cell = cell
        self.cell_type = cell_type
        self.nodes = nn.ModuleList()

        # create nodes name, output, cell_type, scale, summation_function, dim=None)
        for node in range(cell.shape[0]):
            if node == 0:
                self.nodes += [Node(0,
                                    cell_type=self.cell_type,
                                    concatenate=self.cell[node, node],
                                    input_node=(self.layer > 0),
                                    dim=self.in_dim)]
                self.nodes[0].set_post_pad_channels()
            else:
                self.nodes += [Node(node,
                                    cell_type=self.cell_type,
                                    concatenate=self.cell[node, node],
                                    output_node=(node == cell.shape[0] - 1))]

        # create edge parameters
        self.edges = nn.ModuleList()
        edge_queue = indeces(self.cell)

        # create edges and connect them to origin and target nodes
        for origin, target in edge_queue:
            # node summation function
            if origin == target:
                self.nodes[origin].set_post_pad_channels()

                if not cell[origin, target]:
                    if not all([x == self.nodes[origin].dim[0] for x in self.nodes[origin].dim]):
                        raise ValueError("Input Mismatch for Summation Node {}".format(target))
                else:
                    new_channels = sum([x[1] for x in self.nodes[origin].dim])
                    self.nodes[origin].dim = [channel_mod(self.nodes[origin].dim[0], new_channels)]

            # node send attributes
            elif origin > target:
                send_origin, send_target = target, origin
                frac = 1 if cell[origin, target] == 0 else 1 / min(self.in_dim[1], sum(cell[target + 1:, target]))
                self.nodes[send_origin].send_fraction[send_target] = frac

            # edge operations
            elif origin < target and self.cell[origin, target]:
                op = idx_to_op[int(self.cell[origin, target])]
                if verbose:
                    print(origin, target, op)
                new_edge = Edge(target=self.nodes[target],
                                origin=self.nodes[origin],
                                send_frac=self.nodes[origin].send_fraction[target],
                                op=op)
                self.nodes[origin].add_cnx(new_edge, send_frac=self.nodes[origin].send_fraction[target])
                self.edges.append(new_edge)

    def get_out_dim(self):
        # find output dimension of cell
        return self.nodes[-1].dim[0]

    def get_upscale(self):
        # determine scaling ratios of cell
        filters = self.nodes[-1].dim[0][1] / self.in_dim[1]
        dims = self.nodes[-1].dim[0][2] / self.in_dim[2]
        return filters, dims

    def forward(self, x, drop_path, verbose):
        if drop_path:
            preserved = {0, self.nodes[-1].name}
            for i, node in enumerate(self.nodes):
                node_cnx = {}
                if i in preserved:
                    target_names = node.target_names_list
                    cnxs = node.num_targets
                    for target in target_names:
                        rand_val = random.random()
                        if target not in preserved and rand_val > .5 and cnxs > 1:
                            cnxs -= 1
                        else:
                            preserved.add(target)
                            node_cnx[i, target] = True
                out = node.forward(x, node_cnx=node_cnx, verbose=verbose)
        else:
            for i, node in enumerate(self.nodes):
                out = node.forward(x, verbose=verbose)
        return out

    def __repr__(self):
        out = ""
        for i, node in enumerate(self.nodes):
            out += str(node)
        out += "Cell Params: {:,}\n".format(general_num_params(self))
        return out
