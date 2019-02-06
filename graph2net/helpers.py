import math
import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch


# === MODEL HELPERS ====================================================================================================
def general_num_params(model):
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


def namer():
    names = open("graph2net/names.txt", "r").readlines()
    len_names = len(names)
    choices = np.random.randint(0, len_names, 3)
    return " ".join([names[i].strip() for i in choices])


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert (isinstance(size, torch.Size))
    return " x ".join(map(str, size)), int(np.prod(list(map(int, size))))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    out = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    dim, size = pretty_size(obj.size())
                    out += [[str(type(obj).__name__), dim, size]]
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    dim, size = pretty_size(obj.size())
                    out += [[str(type(obj).__name__), dim, size]]
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    out = pd.DataFrame(out, columns=['name', 'dim', 'size'])
    out = out.sort_values(by='size', ascending=False)
    return out


def cell_space(reductions,spacing):
    return ([1]+[0]*spacing)*(reductions-1)+[1]


def max_model_size(cell,data):
    from graph2net.trainers import gen_and_validate
    cell_results = []

    for scale in reversed(range(4, 7)):
        for spacing in range(2, 6):
            for parallel in [True, False]:
                cell_list = [cell, cell] if parallel else [cell]
                model, valid, reason = gen_and_validate(cell_list,
                                                        data,
                                                        scale=scale,
                                                        cell_types=cell_space(5, spacing),
                                                        verbose=False)
                params = model.get_num_params()

                del model
                torch.cuda.empty_cache()

                if valid:
                    cell_results.append([scale, spacing, parallel, params])
            if not valid:
                break
        if len(cell_results) or reason == 'other':
            break

    if len(cell_results):
        cell_results.sort(key=lambda x: (x[3], x[1]), reverse=True)
        return cell_results[0]
    else:
        return False


# === NUMPY HELPERS ====================================================================================================
def indeces(arr):
    size = arr.shape[0]
    out = []
    for i in range(arr.shape[0]):
        out += [(i, i)]
        out += [(x, i) for x in range(i + 1, size)]
        out += [(i, x) for x in range(i + 1, size)]
    return out


def channel_mod(l, value):
    return [x if i != 1 else value for (i, x) in enumerate(l)]


def width_mod(l, by):
    return [x if i not in [2, 3] else int(x / by) for (i, x) in enumerate(l)]


# === I/O HELPERS ======================================================================================================
def eq_string(n):
    return "=" * n


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def log_print(string, end="\n", flush=False):
    print(string, end=end, flush=flush)
    logging.info(string)


def progress_bar(n, length):
    n = int(10 * (n / length))
    out = ""
    out += "=" * n
    out += "-" * (10 - n)
    return out


def curr_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def div_remainder(n, interval):
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    if seconds < 60:
        return "{:.2f} s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{} min, {} s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{:} hrs, {} mins, {} s".format(hours, minutes, seconds)


def loss_plot(epochs, losses):
    plt.plot(range(epochs), losses)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
