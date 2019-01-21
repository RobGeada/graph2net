import copy
import logging
import numpy as np
import pandas as pd
import pickle as pkl
from time import time

from graph2net.trainers import gen_and_validate, full_model_run, accuracy_prediction
from graph2net.graph_generators import gen_cell, swap_mutate
from graph2net.helpers import show_time, namer


# micro/macro predictor function
def macro_predict(row):
    predictor = pkl.load(open('pair_predictors.pkl', "rb"))
    X = np.array([row['acc_preds'],13,row['params'],1,len(row['cell'][0])])
    return predictor['b']+np.dot(X, predictor['m'])


# gene pool creator
def create_gene_pool(size, creation_params, inclusions=[]):
    node_range = creation_params['node_range']
    connectivity_range = creation_params['connectivity_range']
    concat = creation_params['concat']

    cells = []
    for s in range(size):
        print("Generating cell {:^3} of {:^3}...".format(s+1,size), end="\r")
        nodes = np.random.randint(node_range[0], node_range[1])
        connectivity = np.random.uniform(connectivity_range[0], connectivity_range[1])
        cells.append({'cell': gen_cell(nodes, connectivity, concat),
                      "genotype": "n: {}, c: {:.3}".format(nodes, connectivity),
                      "mutations": 0,
                      "loss": 0.,
                      "correct": 0,
                      "adult": False,
                      "name": namer(),
                      "params": 0,
                      "offspring": 0,
                      "lineage": [],
                      "preds": [],
                      "acc_preds": 0,
                      "confidence": 0,
                      "macro_pred":0})
        cells[s]['type'] = "concat" if cells[s]['cell'][-1, -1] else "sum"

    for inclusion in inclusions:
        inclusion['type'] = "concat" if inclusion['cell'][-1, -1] else "sum"
        inclusion['loss'] = 0.
        inclusion['correct'] = 0
        inclusion['mutations'] = 0
        inclusion['adult'] = False
        inclusion['name'] = namer()
        inclusion["params"] = 0
        inclusion['offspring'] = 0
        inclusion['lineage'] = []
        inclusion['preds'] = []
        inclusion["acc_preds"] = 0
        inclusion["confidence"] = 0
        inclusion["macro_pred"] = 0
        cells.append(inclusion)
    return pd.DataFrame(cells)


# run gene pool
def run_petri(df, **kwargs):
    previous_score, best_score = 0, 0
    times = []
    for index, cell in df.iterrows():
        best_score = int(cell['correct']) if int(cell['correct']) > best_score else best_score
        if not cell['adult']:
            start = time()
            start_t = time()

            ave_time = 0 if len(times) == 0 else np.mean(times)
            prefix = "Incubating Cell {}: {}, m: {}, id: {}, t/per: {}s (best: {:005.2f}%, current: ".format(
                index,
                cell['genotype'],
                cell['mutations'],
                cell['name'],
                int(ave_time),
                best_score / len(kwargs['data'][1].dataset) * 100)

            if kwargs.get('verbose', True):
                print(prefix)
            else:
                print(prefix + "00.00%)", end="".join([" "] * 100) + "\r")
                # logging.info(prefix)

            if kwargs['connectivity'] == 'parallel':
                cell_matrices = [cell['cell'], cell['cell']]
            else:
                cell_matrices = [cell['cell']]
            model = gen_and_validate(cell_matrices=cell_matrices,
                                     data=kwargs['data'],
                                     cell_types=kwargs['cell_types'],
                                     scale=kwargs['scale'],
                                     verbose=False)

            if model:
                lr_schedule = {'type': 'cosine',
                               'lr_min': .00001,
                               'lr_max': .01,
                               't_0': kwargs['epochs'],
                               't_mult': 1}
                df.at[index, 'params'] = model.get_num_params()

                loss, correct, preds, acc_preds, confs = full_model_run(model,
                                                                        data=kwargs['data'],
                                                                        epochs=kwargs['epochs'],
                                                                        lr=.01,
                                                                        momentum=.9,
                                                                        weight_decay=1e-4,
                                                                        lr_schedule=lr_schedule,
                                                                        drop_path=True,
                                                                        log=True,
                                                                        track_progress=not kwargs.get('verbose', True),
                                                                        prefix=prefix,
                                                                        verbose=kwargs.get('verbose', True))
                previous_score = max(correct)
                best_score = previous_score if previous_score > best_score else best_score

                best_epoch = np.argmax(correct)
                df.at[index, 'loss'] = -min(loss) if not np.isnan(-min(loss)) else 0.

                #if kwargs.get('predict', False):
                #    print("Setting correctness to pred",acc_pred)
                df.at[index, 'acc_preds'] = acc_preds[-1]
                df.at[index, 'confidence'] = confs[-1]
                df.at[index, 'correct'] = max(correct)
                df.at[index, 'preds'] = preds[best_epoch]
                df.at[index, 'macro_pred'] = macro_predict(df.loc[index])
            else:
                df.at[index, 'loss'] = 0.
                df.at[index, 'acc_preds'] = []
                df.at[index, 'correct'] = 0
                df.at[index, 'preds'] = []
            df.at[index, 'adult'] = True
            if kwargs.get('verbose', True):
                print("\t Time: {}, Corrects: {}".format(show_time(time() - start_t), df.at[index, 'correct']))
            times.append(time()-start)

    return df.sort_values('acc_preds', ascending=False)


def mutate_pool(pool, parents, children, mutation_probability, new_entries, creation_params):
    seed_pool = pool.iloc[:parents]
    seed_pool_sample = seed_pool.sample(n=children, replace=True)

    new_pool = []
    for i,(index, cell) in enumerate(seed_pool_sample.iterrows()):
        print("Mutating cell {:^3} of {:^3}...".format(i + 1, children), end="\r")
        pool.at[index, 'offspring'] += 1
        child_cell = swap_mutate(cell['cell'], mutate_prob=mutation_probability, swap_prob=.5)
        new_pool.append({'cell': child_cell,
                         'type': "concat" if child_cell[-1, -1] else "sum",
                         'genotype': cell['genotype'],
                         'mutations': cell['mutations'] + 1,
                         'loss': 0.,
                         'correct': 0,
                         'adult': False,
                         "name": namer(),
                         "params":0,
                         "offspring": 0,
                         "lineage": [cell['name']] + cell['lineage'],
                         "preds": [],
                         "acc_preds": 0,
                         "confidence": 0,
                         "macro_pred": 0})
    print()
    new_genes = create_gene_pool(new_entries, creation_params=creation_params)
    new_df = pd.DataFrame(new_pool)
    return pool.append(new_df).append(new_genes).reset_index(drop=True)
