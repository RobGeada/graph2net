import numpy as np
import torch
from torchvision import datasets,transforms


# === MODEL HELPERS ====================================================================================================
def general_num_params(model):
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


def eq_string(n):
    return "=" * n


# === DATA HELPERS =====================================================================================================
def load_data(verbose=False):
    batch_size = 256
    image_shape = [32, 32]
    color_channels = 3
    powers = [(2 ** x) * color_channels for x in range(3, 20)]
    data_shape = [batch_size, color_channels] + image_shape
    classes = 10

    if verbose:
        print("Channel Progression:", powers)
        print("Data shape:", data_shape)
        print("Classes:   ", classes)

    train_data = datasets.CIFAR10('data10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize(image_shape),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0,), (1,))]
                                  ))
    test_data = datasets.CIFAR10('data10',
                                 train=False,
                                 download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_shape),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0,), (1,))]
                                 ))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    batches = len(train_loader)
    if verbose:
        print("Factors of", batches, "batches: ", [x for x in range(1, batches) if (batches / x) % 1 == 0])

    return train_loader, test_loader, data_shape, classes
