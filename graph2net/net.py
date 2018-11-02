from graph2net.cell import Cell
from graph2net.ops import Classifier
from graph2net.helpers import *
import sys
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, cell_matrices, cell_types, in_dim, classes, scale):
        super(Net, self).__init__()
        self.cell_matrices = cell_matrices
        self.layers = nn.ModuleList()
           
        # create and join cells
        for layer, cell_type in enumerate(cell_types):
            old_dim = in_dim
            layer_cells = nn.ModuleList()

            for i, cell_matrix in enumerate(cell_matrices):
                new_cell = Cell(in_dim, cell_matrix, cell_type=cell_type, scale=scale)
                new_dim = new_cell.get_out_dim()

                # make sure cells scale tensors correctly
                if len(cell_types) > 1 and old_dim[1] > in_dim[1] and old_dim[2] < new_dim[2]:
                    raise ValueError("Cell doesn't properly scale")

                layer_cells.append(new_cell)

            self.layers.append(layer_cells)
            in_dim = new_dim

        # append linear classifier after all cells
        self.classifier = Classifier(int(np.prod(in_dim[1:])), classes)
        
    # find scaling ratio of component cell
    def get_cell_upscale(self):
        return [[cell.get_upscale() for cell in layer] for layer in self.layers]
                
    def get_num_params(self):
        return general_num_params(self)
            
    def forward(self, x, drop_path, verbose):
        # pass data through cells
        for layer in self.layers:
            x = sum([cell(x, drop_path=drop_path, verbose=verbose) for cell in layer])

        # pass data through classifier
        x = self.classifier(x)
        if verbose:
            sys.exit()
        return x

    def minimal_print(self):
        out = "{} NETWORK CELL {}\n".format(eq_string(50), eq_string(50))
        return out+str(self.layers[0])
        
    def __repr__(self):
        out = ""
        cell_types = ['Normal','Reduction']
        for i, layer in enumerate(self.layers):
            out += "\n{}\n{} LAYER {} {}\n{}\n".format(eq_string(109),eq_string(50), i, eq_string(50),eq_string(109))
            for j, cell in enumerate(layer):
                out += "{} CELL {},{} ({}) {}\n".format(eq_string(40), i,j, cell_types[cell.cell_type],eq_string(40))
                out += str(cell)
            
        out += str(self.classifier)
        out += "{}\nTOTAL PARAMS: {:,}\n".format(eq_string(108), self.get_num_params())
        return out  