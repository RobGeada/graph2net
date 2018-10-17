import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

from graph2net.graph_generators import show_matrix
from graph2net.net import Net

def load_data(verbose):
    batch_size=64
    image_shape = [32,32]
    color_channels = 3
    powers = [(2**x)*color_channels for x in range(3,20)]
    if verbose: print("Channel Progression:",powers)
    data_shape = [batch_size,color_channels]+image_shape
    classes=10
    if verbose: print("Data shape:",data_shape)
    if verbose: print("Classes:   ",classes)

    train_data= datasets.CIFAR10('data10', 
                          train=True, 
                          download=False,
                          transform=transforms.Compose([
                              transforms.Resize(image_shape),
                              transforms.ToTensor(),
                              transforms.Normalize((0,), (1,))]
                          ))
    test_data= datasets.CIFAR10('data10', 
                          train=False,
                          download=False,
                          transform=transforms.Compose([
                              transforms.Resize(image_shape),
                              transforms.ToTensor(),
                              transforms.Normalize((0,), (1,))]
                          ))
                          
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_data,
                                               batch_size=batch_size, 
                                               shuffle=False)

    batches = len(train_loader)
    if verbose:  print("Factors of",batches,"batches: ",[x for x in range(1,batches) if (batches/x)%1==0])

    return train_loader,test_loader,data_shape,classes

#single epoch train function
def train(model, device, train_loader, criterion, optimizer, epoch, validate=False,verbose=False):
    if not validate: print()
    ims = []
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)        
        
        optimizer.zero_grad()
        output = model.forward(data,verbose)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 2 == 0:
            if not validate:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()),
                     end="\n",flush=True)
        if validate: return True

#single epoch validation function
def test(model, device, test_loader):
    model.eval()
    test_loss= 0
    corrects = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output=model.forward(data,verbose=False)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            corrects += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss = np.round(test_loss/len(test_loader.dataset),2)
                
    print('\nTest Loss',test_loss,'Corrects',corrects,"/",len(test_loader.dataset))
    return test_loss
    
#run n epochs of the training and validation
def full_model_test(model,epochs):
    device = torch.device("cpu")
    lr=.05
    momentum=.9
    losses= []
    optimizer = optim.SGD(model.parameters(),lr=.025,momentum=.9,weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        train(model,device, train_loader, criterion, optimizer, epoch)
        losses.append(test(model,device,test_loader))
    return -min(losses)

#run a single batch through the model and perform a single optimization step, just to make sure everything works ok
def model_validate(model,train_loader,verbose):
    print("Validating model...",end="")
    device = torch.device("cpu")
    lr=.05
    momentum=.9
    losses= []
    optimizer = optim.SGD(model.parameters(),lr=.025,momentum=.9,weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    try:
        valid=train(model,device, train_loader, criterion, optimizer, epoch=0, validate=True,verbose=False)
        print("[SUCCESS]")
    except KeyboardInterrupt:
        raise
    except:
        print("\n{} ERROR: PRINTING MODEL DEBUGGING {}".format(eq_string(50),eq_string(50)))
        print(model)
        show_matrix(model.matrix,"Failed Model")
        train(model,device, train_loader, criterion, optimizer, epoch=0, validate=True,verbose=True)

#generate and display model, then validate that it runs correctly 
def gen_and_validate(cell,name="Validating Network",cell_count=None,verbose=True,keep_model=True):
    train_loader,test_loader,data_shape,classes=load_data(verbose=False)
    if verbose: print(cell)
    if verbose: show_matrix(cell,name)

    #initialize and display model
    cell_count = math.floor(math.log(32,2)) if cell_count is None else cell_count
    model = Net(cells=cell_count,matrix=cell,in_dim=data_shape,classes=classes)
    scales = model.get_cell_upscale()[-1]
    params = model.get_num_params()
    
    if scales[0]>2:
        print("Incorrect scaling ({}), skipping".format(scales))
        return params
    
    if verbose: print("Cell Scaling Factors:",scales)
    if verbose: print("Network Parameters: {:,}".format(params))
    if verbose: print(model.minimal_print())

    #validate
    model_validate(model,train_loader,verbose=verbose)
    
    if keep_model:
        return model
    else:
        return params


        