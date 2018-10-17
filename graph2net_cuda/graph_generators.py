import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np

from graph2net.net import Net
from graph2net.ops import *

def gen_cell(nodes,connectivity):
    #generate random cell matrix with $nodes nodes
    #connects node_i to node_j with probabilty $sparsity thresh
    nn = np.zeros((nodes,nodes))
    for i in range(nodes):
        for j in range(nodes):
            if j>i:
                num = np.random.randint(1,13) if np.random.rand() < connectivity else 0
                nn[i,j]=num
            if j>0 and i==nodes-1 and all(nn[:,j]==0):
                nn[0,j]=1
        if i<nodes-1 and all(nn[i,]==0):
            nn[i,-1]=1
    return nn

def build_matrix(pairs):
    #build cell matrix from connectivity pairs
    #input [(0,1,3),(1,2,4)] means node0->node1 via op 3, node1->node2 via op4, etc
    
    nodes = np.max([[x[0],x[1]] for x in pairs])+1
    matrix = np.zeros([nodes,nodes])
    for origin,target,function in pairs:
        if type(function)==str:
            matrix[origin,target]=op_to_idx[function]
        else:
            matrix[origin,target]=function
    return matrix
     
def show_matrix(matrix,name):
    #plot cell DAG
    g = nx.DiGraph()
    for origin,target in zip(*matrix.nonzero()):
        g.add_edge(origin,target,function=idx_to_op[int(matrix[origin,target])])
    pos=graphviz_layout(g, prog='dot')
    
    size=matrix.shape[0]
    plt.figure(figsize=(size*1.5,size*1.5))
    nx.draw_networkx(g,pos)
    edge_labels = nx.get_edge_attributes(g,'function')
    nx.draw_networkx_edge_labels(g,pos=pos,edge_labels=edge_labels,arrows=True)
    plt.title(name)
    plt.show()

def stack_matrix(matrix,cells): 
    dim = matrix.shape[0]
    out = np.zeros([dim*cells-1]*2)
    for cell in range(cells):
        shift = 0 if cell==0 else 1
        a,b=cell*(dim-shift),cell*(dim-shift)+dim
        out[a:b,a:b]=matrix
    return out
    
def cubify(matrix):
    #one hot each element in cell matrix (returns matrix of shape [n_nodes,n_nodes,n_ops])
    out = []
    ops = len(idx_to_op)-1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if j>i:
                row = np.zeros(ops)
                op = int(matrix[i,j])
                if op!=0:
                    row[op-1]=1
                out+= list(row)
    return np.array(out)