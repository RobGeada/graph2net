import logging
import numpy as np
import random
from time import time
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from graph2net.graph_generators import show_cell
from graph2net.net import Net
from graph2net.helpers import *
import warnings


# === EPOCH LEVEL FUNCTIONS ============================================================================================
def adjust_lr(optimizer, by=None, verbose=False):
    for param_group in optimizer.param_groups:
        if by is not None:
            param_group['lr'] *= by
        curr_lr = param_group['lr']
    if verbose:
        log_print("\x1b[31mAdjusting lr to {}\x1b[0m".format(curr_lr))
    return curr_lr


def cosine_anneal_lr(optimizer, lr_min, lr_max, t_0, t, verbose):
    for param_group in optimizer.param_groups:
        curr_lr = lr_min + .5 * (lr_max - lr_min) * (1 + np.cos((t * np.pi / t_0)))
        param_group['lr'] = curr_lr
    if verbose:
        log_print("\n\x1b[31mAdjusting lr to {}\x1b[0m".format(curr_lr))
    return curr_lr


def accuracy_prediction(epoch, corrects, params, parallel, cells, nodes, scale, lr_max, reductions):
    try:
        if reductions == 5:
            predictors = pkl.load(open('pickle_jar/macro_loss_predictors.pkl', "rb"))
            if predictors.get(epoch):
                X = np.array([max(corrects), params, parallel, cells, nodes, scale, lr_max])
                return predictors[epoch]['b'] + np.dot(X, predictors[epoch]['m']), predictors[epoch]['95']
        elif reductions == 2:
            predictors = pkl.load(open('pickle_jar/micro_loss_predictors.pkl', "rb"))
            if predictors.get(epoch):
                up_to_corrects = [max(corrects[:i + 1]) for i in range(epoch + 1)]
                X = np.array(up_to_corrects + [params, reductions, parallel, cells, nodes])
                return predictors[epoch]['b'] + np.dot(X, predictors[epoch]['m']), predictors[epoch]['95']
        return max(corrects), 0.
    except FileNotFoundError:
        warnings.warn("No performance predictor file found in pickle jar!")
        return max(corrects), 0.


# === MODEL SIZE FINDER ================================================================================================
def max_model_size(cell, data, auxiliaries=[None]):
    # iteratively finds the largest valid model that fits in memory for a specific cell
    cell_results = []

    for scale in reversed(range(4, 9)):
        for spacing in range(2, 8):
            clean(verbose=False)
            print("=====VALIDATING {},{}======".format(scale,spacing))
            model, valid, reason = gen_and_validate([cell],
                                                    data,
                                                    scale=scale,
                                                    auxiliaries=auxiliaries,
                                                    cell_types=cell_space(5, spacing),
                                                    verbose=False)

            if valid:
                params = model.get_num_params()
                cell_results.append([scale, spacing, params])
                print("SUCCESS")
            del model
            clean(verbose=False)
            if not valid:
                print("FAIL")
                break
        if len(cell_results) or reason == 'other':
            break
    if len(cell_results):
        cell_results.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return cell_results[0]
    else:
        return False


# === BASE LEVEL TRAIN AND TEST FUNCTIONS===============================================================================
def train(model, device, train_loader, **kwargs):
    if kwargs.get('verbose', False) and not kwargs.get('validate', False):
        log_print("=========================================================================")

    #if kwargs.get('validate', False):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        print("Batch {} Init   | Torch Mem: {:<9} | nvidia-smi: {:<9}".format(batch_idx,mem_stats(),nvidia_smi()))
        data, target = data.to(device), target.to(device)

        kwargs['optimizer'].zero_grad()

        if kwargs['drop_path']:
            drop_path = random.random() > .5
        else:
            drop_path = False
        outputs = model.forward(data,
                                drop_path=drop_path,
                                verbose=False,
                                auxiliary=True)

        loss_f = lambda x: kwargs['criterion'](x, target)
        discount = 1 / len(outputs)
        loss = loss_f(outputs[-1]) + sum([discount * loss_f(x) for x in outputs[:-1]])
        print("      {} pre-bw | Torch Mem: {:<9} | nvidia-smi: {:<9}".format(batch_idx,mem_stats(),nvidia_smi()))
        loss.backward()
        kwargs['optimizer'].step()

        if batch_idx % 10 == 0:
            if kwargs.get('verbose', False) and not kwargs.get('validate', False):
                log_print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    kwargs['epoch'],
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()),
                    end="\r", flush=True)
            elif kwargs.get("logger", None):
                kwargs['logger'].info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    kwargs['epoch'],
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
        if batch_idx>1 and kwargs.get('validate', False):
            return True

    if kwargs.get('verbose', False):
        print()


def test(model, device, test_loader, logger=None, verbose=False):
    model.eval()
    test_loss = 0
    corrects = 0
    preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model.forward(data, drop_path=False, verbose=False)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            corrects += pred.eq(target.view_as(pred)).sum().item()
            preds += output.data.tolist()

    test_loss = np.round(test_loss / len(test_loader.dataset), 4)
    if verbose:
        log_print('Test Loss: {}, Corrects: {}/{}'.format(-1 * test_loss,
                                                          corrects,
                                                          len(test_loader.dataset)))
    if logger:
        logger.info('Test Loss: {}, Corrects: {}/{}'.format(-1 * test_loss,
                                                            corrects,
                                                            len(test_loader.dataset)))
    return test_loss, corrects, preds


# === MODEL VALIDATION= ================================================================================================
# run a single batch through the model and perform a single optimization step, just to make sure everything works ok
def model_validate(model, train_loader, verbose):
    if verbose:
        print("Validating model...", end="")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    t_start = time()
    train(model,
          device,
          train_loader,
          criterion=criterion,
          optimizer=optimizer,
          epoch=0,
          validate=True,
          drop_path=False,
          aux=True,
          verbose=False)
    t_end = time()
    if verbose:
        print("[SUCCESS]")
    epoch_time = (t_end - t_start) * len(train_loader)
    if verbose:
        print("Estimated Timelines:\n\t1 epoch   : {}\n\t64 epochs : {}\n\t128 epochs: {}\n\t512 epochs: {}".format(
            show_time(epoch_time),
            show_time(epoch_time * 64),
            show_time(epoch_time * 128),
            show_time(epoch_time * 512)
        ))
        print("Number of parameters: {:,}".format(model.get_num_params()))


# generate model
def generate(cell_matrices, data, **kwargs):
    train_loader, test_loader, data_shape, classes = data

    if kwargs.get('verbose', False):
        print(cell_matrices)
        show_cell(cell_matrices[0], kwargs.get('name', "Validating Cell"))

    # initialize and display model
    cell_types = kwargs.get('cell_types', [1, 1, 1, 1])
    model = Net(cell_types=cell_types,
                cell_matrices=cell_matrices,
                in_dim=data_shape,
                classes=classes,
                residual_cells=kwargs.get('residual_cells', False),
                auxiliaries=kwargs.get('auxiliaries', [None]),
                scale=kwargs.get('scale', 3))
    try:
        if torch.cuda.is_available():
            model = model.cuda()
    except RuntimeError as e:
        if "CUDA" in str(e):
            del model
            clean(verbose=False)
            return False
        else:
            raise e

    scales = model.get_cell_upscale()[-1]
    params = model.get_num_params()

    if kwargs.get('verbose', False):
        print("Cell Scaling Factors:", scales)
        print(model.minimal_print())

    kwargs.update({"scales": scales,
                   'params': params,
                   'cell_matrices': cell_matrices})
    model.stats = kwargs
    return model


#  generate model, check that it works correctly
def gen_and_validate(cell_matrices, data, **kwargs):
    train_loader, test_loader, data_shape, classes = data
    model = generate(cell_matrices, data, **kwargs)

    # validate
    if model:
        try:
            model_validate(model, train_loader, verbose=kwargs.get('verbose', True))
            model.stats['mem_size'] = mem_stats(False)
        except RuntimeError as e:
            if "CUDA" in str(e):
                del model
                clean(verbose=False)
                return None, False, "memory"
            else:
                raise e
        except Exception as e:
            print("Exception:", e)
            return None, False, "other"
    else:
        return model, False, "generation failure"

    return model, True, None


# === TOP LEVEL TRAINING FUNCTION ======================================================================================
# run n epochs of the training and validation
def full_model_run(model, **kwargs):
    # configure logging
    logger = logging.getLogger("Full_model_run")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('logs/model_history.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)

    if kwargs.get("log", False):
        logger.info("\n\n==========================================================================================")
        logger.info("======================================= NEW MODEL ========================================")
        logger.info("==========================================================================================")
        logger.info("Run started at {}".format(curr_time()))
        logger.info("----------- Model stats ----------- ".format(curr_time()))
        [logger.info("{}: {}".format(k, v)) for k, v in model.stats.items()]
        logger.info("-----------  Run stats  ----------- ".format(curr_time()))
        [logger.info("{}: {}".format(k, v)) for k, v in kwargs.items() if k != 'data']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, data_shape, classes = kwargs['data']

    if kwargs['lr_schedule']['type'] == "cosine":
        t = 0
        lr = kwargs['lr_schedule']['lr_max']
        t_0 = kwargs['lr_schedule']['t_0']
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=kwargs['momentum'],
                          weight_decay=kwargs['weight_decay'],
                          nesterov=True)

    criterion = nn.CrossEntropyLoss()
    losses, corrects, all_preds, epoch_times = [], [], [], []
    train_start = time()
    try:
        acc_preds, confidences = [], []
        for epoch in range(kwargs['epochs']):
            if epoch and not (epoch + 1) % 32:
                torch.save(model.state_dict(), "checkpoints/model_ckpt_{}".format(epoch))
            t_start = time()
            train(model,
                  device,
                  train_loader,
                  criterion=criterion,
                  optimizer=optimizer,
                  epoch=epoch,
                  drop_path=kwargs.get('drop_path', True),
                  logger=logger,
                  verbose=kwargs.get('verbose', False))
            epoch_times.append(time() - t_start)
            mean_epoch_time = np.mean(epoch_times)

            if kwargs.get('verbose', False):
                print("Per epoch time:     {}".format(show_time(mean_epoch_time)))
                print("Est remaining time: {}".format(show_time((kwargs['epochs'] - epoch) * mean_epoch_time)))
            if kwargs.get("log", False):
                logger.info("Per epoch time:     {}".format(show_time(mean_epoch_time)))
                logger.info("Est remaining time: {}".format(show_time((kwargs['epochs'] - epoch) * mean_epoch_time)))

            loss, correct, preds = test(model, device, test_loader, logger=logger, verbose=kwargs.get('verbose', False))

            losses.append(loss)
            corrects.append(correct)

            if np.mean(corrects[-5:]) == 1000 and epoch > 5:
                if kwargs.get('log', False):
                    log_print("TERMINATED OVER-SATURATED MODEL EARLY AT EPOCH {}".format(epoch))
                else:
                    print("Terminating over-saturated model early!")
                return [], np.full(kwargs['epochs'], 1000), [], [], []

            acc_pred, confidence = accuracy_prediction(epoch,
                                                       corrects,
                                                       params=general_num_params(model),
                                                       parallel=len(model.cell_matrices),
                                                       cells=len(model.cell_types),
                                                       nodes=len(model.cell_matrices[0]),
                                                       scale=model.scale,
                                                       lr_max=kwargs['lr_schedule']['lr_max'],
                                                       reductions=sum(model.cell_types))
            acc_preds.append(acc_pred)
            confidences.append(confidence)

            if kwargs.get("log", False):
                logger.info("Prediction: {}".format(int(acc_pred)))

            all_preds.append(preds)

            if kwargs.get('track_progress', False):
                if kwargs['track_progress'] == 'bar':
                    correct_perc = "{:005.2f}%) ".format(correct / len(test_loader.dataset) * 100)
                    print(kwargs['prefix'] + correct_perc + progress_bar(epoch + 1, kwargs['epochs']), end="\r")
                else:
                    correct_perc = "{:005.2f}%".format(correct / len(test_loader.dataset) * 100)
                    print("{} {:>3}/{}: {}, Predicted: {:>4} (±{:>4.0f})".format(kwargs['prefix'],
                                                                                 epoch + 1,
                                                                                 kwargs['epochs'],
                                                                                 correct_perc,
                                                                                 int(acc_pred),
                                                                                 confidence), end="\r")

            if kwargs['lr_schedule']['type'] == 'interval':
                if epoch % kwargs['lr_schedule']['interval'] == 0 and epoch > 0:
                    adjust_lr(optimizer, by=kwargs['lr_schedule']['by'], verbose=kwargs.get('verbose', False))
            elif kwargs['lr_schedule']['type'] == 'schedule':
                if epoch in kwargs['lr_schedule']['schedule']:
                    adjust_lr(optimizer, by=kwargs['lr_schedule']['by'], verbose=kwargs.get('verbose', False))
            elif kwargs['lr_schedule']['type'] == "cosine":
                if t == t_0:
                    t_0 *= kwargs['lr_schedule']['t_mult']
                    t = 0
                    if kwargs.get('verbose', False):
                        print("\n\x1b[31mRestarting Learning Rate, setting new cycle length to {}\x1b[0m".format(t_0))
                    if kwargs.get("log", False):
                        logger.info("Restarting Learning Rate, setting new cycle length to {}\x1b[0m".format(t_0))
                cosine_anneal_lr(optimizer,
                                 lr_min=kwargs['lr_schedule']['lr_min'],
                                 lr_max=kwargs['lr_schedule']['lr_max'],
                                 t_0=t_0,
                                 t=t,
                                 verbose=kwargs.get('verbose', False))
                t += 1

    except KeyboardInterrupt as e:
        if kwargs.get('log', False):
            logger.info("TERMINATED EARLY AT EPOCH {}".format(epoch))
        if kwargs.get('verbose', False):
            print("Terminating early...")
    except Exception as e:
        del model
        clean(verbose=False)
        raise e

    if kwargs.get('log', False):
        logger.info("----------- Completion Stats ----------- ")
        logger.info("Run started at {}".format(curr_time()))
        logger.info("----------- Model stats ----------- ".format(curr_time()))
        [logger.info("{}: {}".format(k, v)) for k, v in model.stats.items()]
        logger.info("-----------  Run stats  ----------- ".format(curr_time()))
        [logger.info("{}: {}".format(k, v)) for k, v in kwargs.items() if k != 'data']
        logger.info("Run finished at {}".format(curr_time()))
        logger.info("Time taken:   {}".format(show_time(time() - train_start)))

        if corrects:
            logger.info("Max corrects: {}/{} at epoch {}".format(
                max(corrects),
                len(test_loader.dataset),
                np.argmax(corrects)))
        else:
            logger.info("Run terminated before a single epoch finished")

    return np.array(losses), np.array(corrects), np.array(all_preds), acc_preds, confidences
