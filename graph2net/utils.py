import math
import numpy as np

from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from graph2net.graph_generators import show_matrix
from graph2net.net import Net


# === EPOCH LEVEL FUNCTIONS ============================================================================================
def adjust_lr(optimizer, by=None, set_to=None):
    for param_group in optimizer.param_groups:
        if set_to is not None:
            param_group['lr'] = set_to
        elif by is not None:
            param_group['lr'] *= by
        curr_lr = param_group['lr']
    print("Adjusting lr to", curr_lr)
    return curr_lr


# === BASE LEVEL TRAIN AND TEST FUNCTIONS===============================================================================
def train(model, device, train_loader, criterion, optimizer, epoch, validate=False, verbose=False):
    if not validate: print()
    ims = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model.forward(data, verbose)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            if not validate:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()),
                    end="\r", flush=True)
        if validate: return True
    print()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    corrects = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model.forward(data, verbose=False)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            corrects += pred.eq(target.view_as(pred)).sum().item()

    test_loss = np.round(test_loss / len(test_loader.dataset), 2)

    print('\nTest Loss', test_loss, 'Corrects', corrects, "/", len(test_loader.dataset))
    return test_loss

# === MODEL VALIDATION= ================================================================================================
# run a single batch through the model and perform a single optimization step, just to make sure everything works ok
def model_validate(model, train_loader, verbose):
    print("Validating model...", end="")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=.025, momentum=.9, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()

    train(model, device, train_loader , criterion, optimizer, epoch=0, validate=True, verbose=False)
    print("[SUCCESS]")


# generate and display model, then validate that it runs correctly
def gen_and_validate(cell, name="Validating Network", data=None,cell_count=None, verbose=True, keep_model=True):
    train_loader, test_loader, data_shape, classes = data
    if verbose: print(cell)
    if verbose: show_matrix(cell, name)

    # initialize and display model
    cell_count = math.floor(math.log(32, 2)) if cell_count is None else cell_count
    model = Net(cells=cell_count, matrix=cell, in_dim=data_shape, classes=classes)
    if torch.cuda.is_available():
        model = model.cuda()
    scales = model.get_cell_upscale()[-1]
    params = model.get_num_params()

    if scales[0] > 2:
        print("Incorrect scaling ({}), skipping".format(scales))
        return params

    if verbose: print("Cell Scaling Factors:", scales)
    if verbose: print("Network Parameters: {:,}".format(params))
    if verbose: print(model.minimal_print())

    # validate
    model_validate(model, train_loader, verbose=verbose)

    return model if keep_model else params

# === TOP LEVEL TRAINING FUNCTION ======================================================================================
# run n epochs of the training and validation
def full_model_run(model, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, data_shape, classes = kwargs['data']
    losses = []
    optimizer = optim.SGD(model.parameters(),
                          lr=kwargs['lr'],
                          momentum=kwargs['momentum'],
                          weight_decay=kwargs['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(kwargs['epochs']):
        t_start = time()
        train(model, device, train_loader, criterion, optimizer, epoch)
        print("Epoch Time:", time() - t_start)
        losses.append(test(model, device, test_loader))

        if epoch in kwargs['lr_schedule']:
            adjust_lr(optimizer, by=.1)
    return -min(losses)
