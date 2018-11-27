from graph2net.cell import Cell
from graph2net.ops import Classifier, Identity
from graph2net.helpers import *
import sys
import torch
import torch.nn as nn

# torch.backends.cudnn.deterministic=True


class Net(nn.Module):
    def __init__(self, cell_matrices, cell_types, in_dim, classes, scale, residual_cells=False):
        super(Net, self).__init__()
        self.cell_matrices = cell_matrices
        self.residual_cells = residual_cells
        self.layers = nn.ModuleList()
        self.padders = nn.ModuleList()

        # create and join cells
        for layer, cell_type in enumerate(cell_types):
            old_dim = in_dim
            sublayer = nn.ModuleList()
            for cell_matrix in cell_matrices:
                new_cell = Cell(in_dim, cell_matrix, cell_type=cell_type, scale=scale)
                new_dim = new_cell.get_out_dim()

                # make sure cells scale tensors correctly
                if len(cell_types) > 1 and old_dim[1] > in_dim[1] and old_dim[2] < new_dim[2]:
                    raise ValueError("Cell doesn't properly scale")

                sublayer.append(new_cell)

            stride = 2 if old_dim[2] > new_dim[2] else 1
            self.padders.append(Identity(old_dim[1], new_dim[1], stride))
            self.layers.append(sublayer)
            in_dim = new_dim

        # append linear classifier after all cells
        self.classifier = Classifier(int(np.prod(in_dim[1:])), classes)
        
    # find scaling ratio of component cell
    def get_cell_upscale(self):
        return [layer[0].get_upscale() for layer in self.layers]
                
    def get_num_params(self):
        return general_num_params(self)
            
    def forward(self, x, drop_path, verbose):
        # pass data through cells

        for i, layer in enumerate(self.layers):
            cell_out = sum([cell(x, drop_path=drop_path, verbose=verbose) for cell in layer])
            if self.residual_cells:
                residual = self.padders[i](x)
                x = cell_out + residual
            else:
                x = cell_out

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
        cell_types = ['Normal', 'Reduction']
        for i, layer in enumerate(self.layers):
            out += "{} LAYER {} {}".format(eq_string(50),i,eq_string(50))
            for j, cell in enumerate(layer):
                out += "{} CELL {},{} ({}) {}\n".format(eq_string(40), i, j, cell_types[cell.cell_type],eq_string(40))
                out += str(cell)

        out += str(self.classifier)
        out += "{}\nTOTAL PARAMS: {:,}\n".format(eq_string(108), self.get_num_params())
        return out