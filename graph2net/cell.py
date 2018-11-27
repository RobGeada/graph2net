from graph2net.node import Node
from graph2net.ops import *
from graph2net.edge import Edge
from graph2net.helpers import general_num_params

import random
# random.seed(7)


class Cell(nn.Module):
    def __init__(self, in_dim, cell, cell_type=0, scale=3, verbose=False):
        super(Cell, self).__init__()
        self.in_dim = in_dim
        self.cell_type = cell_type
        self.nodes = nn.ModuleList()

        # create nodes
        for node in range(cell.shape[0]):
            if node == 0:
                self.nodes += [Node(0, in_dim)]
            else:
                self.nodes += [Node(node)]

        # create edge parameters
        self.edges = nn.ModuleList()
        cell_width = cell.shape[1]
        edge_queue = list(zip(*cell.nonzero()))

        # create edges and connect them to origin and target nodes
        for origin, target in edge_queue:
            op = idx_to_op[int(cell[origin, target])]
            if verbose:
                print(origin, target, op)
            new_edge = Edge(target=self.nodes[target],
                            origin=self.nodes[origin],
                            op=op)
            self.nodes[origin].add_cnx(new_edge,
                                       o_pos=origin / (cell_width - 1),
                                       t_pos=target / (cell_width - 1),
                                       cell_type=self.cell_type,
                                       scale=scale)
            self.edges.append(new_edge)

    def get_out_dim(self):
        # find output dimension of cell
        return self.nodes[-1].dim

    def get_upscale(self):
        # determine scaling ratios of cell
        filters = self.nodes[-1].dim[1] / self.in_dim[1]
        dims = self.nodes[-1].dim[2] / self.in_dim[2]
        return filters, dims

    def forward(self, x, drop_path, verbose):
        if drop_path:
            preserved = {0,  self.nodes[-1].name}
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
                            node_cnx[i,target] = True
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
