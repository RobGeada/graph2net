import torch
import torch.nn as nn

from graph2net.node import Node
from graph2net.ops import *
from graph2net.edge import Edge
from graph2net.helpers import *

class Cell(nn.Module):
    def __init__(self,in_dim,matrix,reduce=False,verbose=False,to_probe=None):
        super(Cell, self).__init__()
        self.in_dim = in_dim
        self.nodes = nn.ModuleList()
        
        #create nodes
        for node in range(matrix.shape[0]):
            if node==0:
                self.nodes += [Node(0,in_dim)]
            else:
                self.nodes += [Node(node)]
        
        #create edge parameters
        self.edges = nn.ModuleList()
        matrix_width = matrix.shape[1]
        edgeQueue = list(zip(*matrix.nonzero()))
        
        #create edges and connect them to origin and target nodes
        for origin,target in edgeQueue:
            op = idx_to_op[int(matrix[origin,target])]            
            if verbose:
                print(origin,target,op)
            new_edge = Edge(target=self.nodes[target],op=op)
            self.nodes[origin].add_cnx(new_edge,o_pos=origin/(matrix_width-1),t_pos=target/(matrix_width-1))
            self.edges.append(new_edge)
            
    def get_out_dim(self):
        #find output dimension of cell
        return self.nodes[-1].dim
    
    def get_upscale(self):
        #determine scaling ratios of cell
        filters =  self.nodes[-1].dim[1]/self.in_dim[1]
        dims = self.nodes[-1].dim[2]/self.in_dim[2]
        return filters,dims
    
    def forward(self,x,verbose):
        for i,node in enumerate(self.nodes):
            if i==0:
                #if node is the input node in cell, pass in cell input
                out=node.forward(x,verbose=verbose)
            else:
                #otherwise, nodes will receive input tensors from prior nodes
                out=node.forward(verbose=verbose)
        return out

    def __repr__(self):
        out = ""
        for i,node in enumerate(self.nodes):
            out+=str(node)
        return out