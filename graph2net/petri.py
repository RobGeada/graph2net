import copy
import logging
import numpy as np
import pandas as pd
from time import time

from graph2net.trainers import gen_and_validate, full_model_run
from graph2net.graph_generators import gen_cell
from graph2net.helpers import show_time, eq_string, namer
from graph2net.ops import idx_to_op


def create_gene_pool(size, node_range, connectivity_range,inclusions=[]):
    cells = []
    for s in range(size):
        nodes = np.random.randint(node_range[0], node_range[1])
        connectivity = np.random.uniform(connectivity_range[0], connectivity_range[1])
        cells.append({'cell': gen_cell(nodes, connectivity),
                      "genotype": "n: {}, c: {:.3}".format(nodes, connectivity),
                      "mutations": 0,
                      "loss": 0.,
                      "correct": 0,
                      "adult": False,
                      "name": namer(),
                      "offspring": 0,
                      "lineage":[]})

    for inclusion in inclusions:
        inclusion['loss'] = 0.
        inclusion['correct'] = 0
        inclusion['mutations'] = 0
        inclusion['adult'] = False
        inclusion['name'] = namer()
        inclusion['offspring'] = 0
        inclusion['lineage'] = []
        cells.append(inclusion)
    return pd.DataFrame(cells)


def run_petri(df, **kwargs):
    previous_score, best_score = 0, 0
    for index, cell in df.iterrows():
        if not cell['adult']:
            start_t = time()

            prefix = "Incubating Cell {}: {}, m: {}, id: {} (previous: {:.2f}%, best: {:.2f}%)".format(
                index,
                cell['genotype'],
                cell['mutations'],
                cell['name'],
                previous_score / len(kwargs['data'][1].dataset),
                best_score / len(kwargs['data'][1].dataset))

            if kwargs.get('verbose',True):
                print(prefix)
            else:
                print(prefix+" ----------", end="".join([" "]*100)+"\r")
                logging.info(prefix)

            model = gen_and_validate(cell_matrices=[cell['cell']],
                                     data=kwargs['data'],
                                     cell_types=kwargs['cell_types'],
                                     scale=kwargs['scale'],
                                     verbose=False)

            if model:
                lr_schedule = {'type': 'cosine',
                               'lr_min': .00001,
                               'lr_max': .01,
                               't_0': 10,
                               't_mult': 1}

                loss, correct = full_model_run(model,
                                               data=kwargs['data'],
                                               epochs=kwargs['epochs'],
                                               lr=.01,
                                               momentum=.9,
                                               weight_decay=1e-4,
                                               lr_schedule=lr_schedule,
                                               drop_path = True,
                                               log=False,
                                               track_progress=not kwargs.get('verbose',True),
                                               prefix=prefix,
                                               verbose=kwargs.get('verbose',True))
                previous_score = max(correct)
                best_score = previous_score if previous_score > best_score else best_score

                df.at[index,'loss'] = -min(loss) if not np.isnan(-min(loss)) else 0.
                df.at[index,'correct'] = max(correct)
            else:
                df.at[index, 'loss'] = 0.
                df.at[index, 'correct'] = 0
            df.at[index,'adult'] = True
            if kwargs.get('verbose', True):
                print("\t Time: {}, Corrects: {}".format(show_time(time() - start_t),df.at[index,'correct']))

    return df.sort_values('correct',ascending=False)


def mutate_cell(cell,mutation_probability):
    mutated_cell = copy.copy(cell)
    changes = []
    nodes = mutated_cell.shape[0]
    for i in range(nodes):
        for j in range(nodes):
            if j > i:
                if not mutated_cell[i][j] and np.random.rand() < mutation_probability:
                    new_op = np.random.randint(1, len(idx_to_op))
                    changes.append([i,j,mutated_cell[i,j],new_op,"Insertion"])
                    mutated_cell[i,j] = new_op
                elif mutated_cell[i][j]:
                    if np.random.rand() < mutation_probability:
                        if np.random.rand() > .5:
                            changes.append([i,j,mutated_cell[i,j],0,"Deletion"])
                            mutated_cell[i, j] = 0
                        else:
                            new_op = np.random.randint(1, len(idx_to_op))
                            changes.append([i, j, mutated_cell[i, j], new_op, "Alteration"])
                            mutated_cell[i, j] = new_op

            if j > 0 and i == nodes - 1 and all(mutated_cell[:, j] == 0):
                changes.append([0,j,mutated_cell[i,j],1,"Restorative"])
                mutated_cell[0, j] = 1
        if i < nodes - 1 and all(mutated_cell[i,] == 0):
            changes.append([i, -1, mutated_cell[i,j],1,"Restorative"])
            mutated_cell[i, -1] = 1

    change_log = []
    for i,j,old_op,new_op,change_type in changes:
        change_log.append("{}: {}->{} at {},{}".format(change_type,
                                                       idx_to_op[old_op],
                                                       idx_to_op[new_op],
                                                       i,j))
    return mutated_cell


def mutate_pool(pool, reproduction_thresh, mutation_probability):
    parent_count = int(len(pool) * reproduction_thresh)
    child_count = len(pool) - parent_count

    seed_pool = pool.iloc[0:parent_count]
    seed_pool_sample = seed_pool.sample(n=child_count, replace=True)

    new_pool=[]
    for index, cell in seed_pool_sample.iterrows():
        pool.at[index,'offspring'] += 1
        child_cell = mutate_cell(cell['cell'], mutation_probability=mutation_probability)
        new_pool.append({'cell': child_cell,
                         'genotype': cell['genotype'],
                         'mutations': cell['mutations']+1,
                         'loss': 0.,
                         'correct': 0,
                         'adult': False,
                         "name": namer(),
                         "offspring": 0,
                         "lineage": [cell['name']]+cell['lineage']})
    new_df = pd.DataFrame(new_pool)
    return pool.append(new_df).reset_index(drop=True)
