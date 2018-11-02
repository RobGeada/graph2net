import math
import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt


# === MODEL HELPERS ====================================================================================================
def general_num_params(model):
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


def namer():
    names =open("graph2net/names.txt", "r").readlines()
    len_names = len(names)
    choices = np.random.randint(0,len_names,3)
    return " ".join([names[i].strip() for i in choices])


# === I/O HELPERS ======================================================================================================
def eq_string(n):
    return "=" * n


def log_print(string, end="\n",flush=False):
    print(string, end=end,flush=flush)
    logging.info(string)


def progress_bar(n,length):
    n = int(10*(n/length))
    out = ""
    out += "=" * n
    out += "-"*(10-n)
    return out


def curr_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def div_remainder(n,interval):
    factor = math.floor(n/interval)
    remainder = int(n-(factor*interval))
    return factor,remainder


def show_time(seconds):
    if seconds < 60:
        return "{:.2f} s".format(seconds)
    elif seconds < (60 * 60):
        minutes,seconds = div_remainder(seconds,60)
        return "{} min, {} s".format(minutes,seconds)
    else:
        hours,seconds   = div_remainder(seconds,60*60)
        minutes,seconds = div_remainder(seconds,60)
        return "{:} hrs, {} mins, {} s".format(hours,minutes,seconds)


def loss_plot(epochs,losses):
    plt.plot(range(epochs), losses)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()










