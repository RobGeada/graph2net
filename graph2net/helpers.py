import math
import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch
import gc
import subprocess


# === MODEL HELPERS ====================================================================================================
def general_num_params(model):
    # return number of differential parameters of input model
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


def namer():
    # generate random tripled-barrelled name to track evolved models
    names = open("graph2net/names.txt", "r").readlines()
    len_names = len(names)
    choices = np.random.randint(0, len_names, 3)
    return " ".join([names[i].strip() for i in choices])


def cell_space(reductions,spacing):
    # spaces out reductions evenly amongst spacings
    # returns array with r repeating groups of s spaces
    return ([0]*spacing+[1])*(reductions)
    # return ([1]+[0]*spacing)*(reductions-1)+[1]


# === TORCH MEMORY HELPERS =============================================================================================
def mem_stats(human_readable=True):
    # returns current allocated torch memory
    if human_readable:
        return sizeof_fmt(torch.cuda.memory_cached())
    else:
        return int(torch.cuda.memory_cached())

def nvidia_smi():
    p = subprocess.Popen('nvidia-smi', stdout=subprocess.PIPE)
    msg,err = p.communicate()
    msg = msg.decode('utf-8')
    return [x.split()[-2] for x in msg.split("\n") if 'python' in x][0]

def clean(model=None, name=None, verbose=True):
    if verbose:
        print('\nCleaning up at {}...'.format(name))
        print('Pre:', mem_stats())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print('Post:', mem_stats())


def pretty_size(size):
    # pretty prints a torch.Size object
    assert (isinstance(size, torch.Size))
    return " x ".join(map(str, size)), int(np.prod(list(map(int, size))))


def dump_tensors(gpu_only=True):
    # Prints a list of the Tensors being tracked by the garbage collector.
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
    out = out.groupby(["name","dim","size"]).size().reset_index(name='counts')
    out['net_size']=out['size']*out['counts']
    out = out.sort_values(by='net_size', ascending=False)
    return out


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


def top_n(arr):
    # return indeces that correspond to highest values of list
    return sorted(enumerate(arr),key=lambda x: x[1],reverse=True)


def np_raw_print(arr):
    print("= np.array([{}])".format(",\n            ".join([str([i for i in j]) for j in arr])))


# === I/O HELPERS ======================================================================================================
def eq_string(n):
    # prints n equals signs, for spacing output strings
    return "=" * n


def sizeof_fmt(num, suffix='B'):
    # turns bytes object into human readable
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.2f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.2f%s%s" % (num, 'Yi', suffix)


def log_print(string, end="\n", flush=False):
    # prints and logs given string
    print(string, end=end, flush=flush)
    logging.info(string)


def progress_bar(n, length):
    # creates progress bar string given progress and total goal
    n = int(10 * (n / length))
    out = ""
    out += "=" * n
    out += "-" * (10 - n)
    return out


def curr_time():
    # returns current time string
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    # show amount of time as human readable
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
    # labeled loss plot
    plt.plot(range(epochs), losses)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
