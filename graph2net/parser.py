import copy

import numpy as np
import pandas as pd
import pickle as pkl
import re
import os 


def line_format(line,add_quotes=True):
    if ("INFO" not in line or 'cell:' in line) and "[" in line and "]" in line:
        if "," not in line:
            if "." in line:
                line = re.sub('(?<=\d)\.+',",",line)
            else:
                line = re.sub('(?<=\d)\s+',",",line)
    if "torch.utils.data" in line:
        line = line.replace("<","'").replace(">","'")

    line = line.replace("\n","").replace(" ","").replace("array","np.array").replace("\e[1m","").replace("\e[21mN","")
    if add_quotes:
        if ":" in line:
            name,val = line.split(":",1)
            if name=='prefix':
                return '"{}":"{}"'.format(name.strip(),val)
            else:
                return '"{}":{}'.format(name.strip(),val)
    return line


def general_stats(line,current,start_cond,end_cond,stats_str):
    if start_cond in line:
        current = start_cond
        stats_str = line_format(line.split("stats:")[-1],False)
    elif current == start_cond:
        if end_cond in line:
            current = ''
            return True,"{}".format(stats_str),current

        else:
            stats_str += '{}'.format(line_format(line,False))
    return False,stats_str,current


def specific_stats(line,current,start_cond,end_cond,stats_str=""):
    #print(current,[(x,line,x in line) for x in end_cond])
    if start_cond in line:
        current = start_cond
        stats_str = ""
    
    elif current == start_cond:
        if 'TERMINATED' in line:
            pass
        elif any([x in line for x in end_cond]):
            current = ''
            return True,"{{{}}}".format(stats_str.rsplit(",",1)[0]),current
        elif line[-1]==",":
            #print("pre add ",stats_str)
            stats_str += '{}'.format(line_format(line.split("run:")[-1]))
            #print("post add",stats_str)
        else:
            #print("pre add ",stats_str)
            stats_str += '{}, '.format(line_format(line.split("run:")[-1]))
            #print("post add",stats_str)
    return False,stats_str,current


def local_exec(exec_str):
    ldict = {}
    exec("s={}".format(exec_str),globals(),ldict)
    return ldict['s']


def parse(last=False):
    log1 = open("logs/old_model_testbed.log","r").read()
    log2 = open("logs/model_testbed.log","r").read()
    log = log1+log2
    outs = log.split("NEW MODEL ")[1:]
    
    runs = []
    if last:
        outs = outs[-1:]
    for i,out in enumerate(outs):
        print(i,end="\r")
        curve = []
        predictions = []
        current = None
        time_taken = None
        max_val = None
        early_terminate=False
        stats_str = ""
        epoch_time = None
        run_date = None
        curve_update = False

        general_model_stats,general_run_stats,specific_model_stats,specific_run_stats = {},{},{},{}
        #print("--LOOP ENTER--")
        for line in out.split('\n'):
            if 'MiB' in line:
                continue
            if line=="" or ('Loss' in line and 'Corrects' not in line and 'Epoch: 0' not in line):
                curve_update=False
                pass
            else:
                finished,stats_str,current = general_stats(line,current,'Model stats:','Run stats:',stats_str)
                if finished:
                    general_model_stats = local_exec(stats_str)
                    stats_str, finished = "",False

                finished,stats_str,current = general_stats(line,current,'Run stats:','=====',stats_str)
                if finished:
                    general_run_stats= local_exec(stats_str)
                    stats_str, finished = "",False

                finished,stats_str,current = specific_stats(line,current,'-- Model stats',[' Run stats '],stats_str)
                if finished:
                    specific_model_stats= local_exec(stats_str)
                    stats_str, finished = "",False

                if not specific_run_stats:
                    finished,stats_str,current = specific_stats(line,current,' Run stats ',['====',"Per epoch","Run finished","Train Epoch: 0"],stats_str)
                    if finished:
                        specific_run_stats= local_exec(stats_str)
                        stats_str, finished = "",False

                #print(line,curve_update)
                if "Corrects" in line and not curve_update:
                    curve.append(int(line.split(":")[-1].split("/")[0]))
                    curve_update = True
                if "Prediction" in line:
                    if "[[" in line:
                        predictions.append(float(line.split("[[")[-1].split("]]")[0]))
                    elif "," in line:
                        predictions.append(float(line.split(":")[-1].split(",")[0]))
                    else:
                        predictions.append(float(line.split(":")[-1]))
                if "Time taken" in line:
                    time_taken = line.split(":")[-1]
                if "terminated" in line.lower():
                    early_terminate=True
                if 'Run started' in line:
                    run_date = line.split("at ")[-1]
                if 'Max corrects' in line:
                    max_val = int(line.split(":")[-1].split("/")[0])
                if 'Per epoch time' in line:
                    epoch_time = line.split(":")[-1].strip()
        #print("--LOOP EXIT--")

        stats = {}
        if general_run_stats:
            stats.update(general_run_stats)
        if general_model_stats:
            stats.update(general_model_stats)
        if specific_run_stats:
            stats.update(specific_run_stats)
        if specific_model_stats:
            stats.update(specific_model_stats)

        for key in stats.keys():
            new_key = key.strip().replace('matrix','matrices')
            #print(key)
            if key=='cell':
                new_key='cell_matrices'
            if key=='momemtum':
                new_key = 'momentum'
            if key=='macro_auxillaries':
                new_key = 'macro_auxiliaries'
            if new_key!=key:
                stats[new_key] = stats.pop(key)

        stats['curve']=curve
        stats['predictions']=predictions
        stats['epoch']=len(curve)
        stats['time_taken']=time_taken
        stats['early_terminate']=early_terminate
        stats['per_epoch_time']=epoch_time
        if not max_val:
            stats['max']=max(curve) if len(curve)>0 else None
        else:
            stats['max']=max_val
        stats['run_date']=run_date
        runs.append(stats)

    run_stats = pd.DataFrame(runs)
    if not last:
        run_stats.to_pickle('pickle_jar/run_stats.pkl')
    return run_stats