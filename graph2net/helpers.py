import numpy as np

def general_num_params(model):
  return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    
def eq_string(n):
  return "="*n