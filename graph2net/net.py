from graph2net.cell import Cell
from graph2net.ops import Classifier, MinimumIdentity, initializer
from graph2net.helpers import *
import sys
import torch
import torch.nn as nn


def pp(i):
    return "{:.2f}".format(i)


def list_pp(l):
    return [pp(x) for x in l]


class Net(nn.Module):
    def __init__(self, cell_matrices, cell_types, in_dim, classes, scale, auxiliaries=None, residual_cells=False):
        super().__init__()
        self.cell_matrices = cell_matrices
        self.residual_cells = residual_cells
        self.scale = scale
        self.layers = nn.ModuleList()
        self.padders = nn.ModuleList()
        self.towers = nn.ModuleDict()
        self.auxiliary_idx = auxiliaries
        self.cell_types = cell_types

        # initial scaler
        powers = [(2 ** x) for x in range(scale, 20)]
        self.pre_init_dim = in_dim
        self.initializer_scale = [x for x in powers if x > self.pre_init_dim[1]][0]
        self.initializer = initializer(c_in=in_dim[1], c_out=self.initializer_scale)
        self.post_init_dim = channel_mod(self.pre_init_dim, self.initializer_scale)

        # create and join cells
        cell_in_dim = self.post_init_dim
        reduction_idx = -1
        for layer, cell_type in enumerate(cell_types):
            old_dim = cell_in_dim
            sublayer = nn.ModuleList()
            reduction_idx = reduction_idx + 1 if cell_type == 1 else reduction_idx
            for cell_matrix in cell_matrices:
                new_cell = Cell(layer, cell_in_dim, cell_matrix, cell_type=cell_type)
                new_dim = new_cell.get_out_dim()

                # make sure cells scale tensors correctly
                if len(cell_types) > 1 and old_dim[1] > cell_in_dim[1] and old_dim[2] < new_dim[2]:
                    raise ValueError("Cell doesn't properly scale")
                sublayer.append(new_cell)

            stride = 2 if old_dim[2] > new_dim[2] else 1
            self.padders.append(MinimumIdentity(old_dim[1], new_dim[1], stride))
            self.layers.append(sublayer)
            cell_in_dim = new_dim

            if cell_type and reduction_idx in self.auxiliary_idx:
                self.towers[str(layer)] = Classifier(int(np.prod(cell_in_dim[1:])),
                                                     classes,
                                                     "Auxiliary {}".format(reduction_idx))

        # append linear classifier after all cells
        self.global_pooling = nn.AdaptiveAvgPool2d(cell_in_dim[1:][-1])
        self.towers['Classifier'] = Classifier(int(np.prod(cell_in_dim[1:])), classes)

    # find scaling ratio of component cell
    def get_cell_upscale(self):
        return [layer[0].get_upscale() for layer in self.layers]

    def get_num_params(self):
        return general_num_params(self)

    def forward(self, x, drop_path, verbose, auxiliary=False):
        # pass data through initializer
        try:
            outputs = []
            x = self.initializer(x)

            # pass data through cells
            for i, layer in enumerate(self.layers):
                cell_out = sum([cell(x, drop_path=drop_path, verbose=verbose) for cell in layer])
                if self.residual_cells:
                    residual = self.padders[i](x)
                    x = cell_out + residual
                    del residual
                else:
                    x = cell_out

                if auxiliary and str(i) in self.towers.keys():
                    outputs.append(self.towers[str(i)](x))

            # pass data through classifier
            x = self.global_pooling(x)
            class_out = self.towers['Classifier'](x)
            outputs.append(class_out)

            if verbose:
                sys.exit()
            return outputs if auxiliary else outputs[0]
        except RuntimeError as e:
            if "CUDA" in str(e):
                clean(verbose=False)
            raise e
        except KeyboardInterrupt as e:
            clean(verbose=False)
            raise e

    def minimal_print(self):
        out = "{}{:^15}{}\n".format(eq_string(50), "NETWORK CELL", eq_string(50))
        return out + str(self.layers[0][0])

    def __repr__(self):
        out = ""
        cell_types = ['Normal', 'Reduct']
        out += "{}{:^15}{}\n".format(eq_string(50), "INITIALIZER", eq_string(50))
        out += "Initializer: {:<18} -> {:<18} {:,} params)\n".format(str(self.pre_init_dim),
                                                                     str(self.post_init_dim),
                                                                     general_num_params(self.initializer))
        out += ""
        for i, layer in enumerate(self.layers):
            out += "{} {:^12}{}\n".format(eq_string(51), "LAYER {:>2}".format(i), eq_string(51))
            for j, cell in enumerate(layer):
                out += "{} CELL {},{} ({}) {}\n".format(eq_string(20), i, j, cell_types[cell.cell_type], eq_string(12))
                out += str(cell)

        for i, aux in self.towers.items():
            if i is not 'Classifier':
                out += "{} after layer {}\n".format(aux, i)
            else:
                out += "{}\n".format(aux)
        out += "{}\nTOTAL PARAMS: {:,}\n".format(eq_string(108), self.get_num_params())
        return out
