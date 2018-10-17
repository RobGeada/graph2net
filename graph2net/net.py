import torch.nn as nn

from graph2net.cell import Cell
from graph2net.ops import Classifier
from graph2net.helpers import *

class Net(nn.Module):
    def __init__(self,cells,matrix,in_dim,classes):
        super(Net, self).__init__()
        self.matrix=matrix
        self.cells = nn.ModuleList()
           
        #create and join cells
        for cell in range(cells):
            old_dim = in_dim
            new_cell = Cell(in_dim,matrix)
            in_dim = new_cell.get_out_dim()
            
            #make sure cells scale tensors correctly
            if cells>1 and old_dim[1]>=in_dim[1] and old_dim[2]<=in_dim[2]:
                raise ValueError("Cell doesn't properly scale")
                
            self.cells.append(new_cell)
        
        #append linear classifier after all cells
        self.classifier = Classifier(int(np.prod(in_dim[1:])),classes)
        
    #find scaling ratio of component cell
    def get_cell_upscale(self):
        return [cell.get_upscale() for cell in self.cells]
                
    def get_num_params(self):
        return general_num_params(self)
            
    def forward(self,x,verbose):
        #pass data through cells
        for cell in self.cells:
            x = cell(x,verbose)
        #pass data through classifier
        x = self.classifier(x)
        if verbose:
            sys.exit()
        return x

    def minimal_print(self):
        out = "{} NETWORK CELL {}\n".format(eq_string(50),eq_string(50))
        return out+str(self.cells[0])
        
    def __repr__(self):
        out = ""
        for i,cell in enumerate(self.cells):
            out+="{} CELL {} {}\n".format(eq_string(50),i,eq_string(50))
            out+=str(cell)
            
        out+=str(self.classifier)
        out+="{}\nTOTAL PARAMS: {:,}\n".format(eq_string(108),self.get_num_params())
        return out  