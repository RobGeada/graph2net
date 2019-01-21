import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from graph2net.ops import *
from graph2net.helpers import *
import random

powers = [2 ** x for x in range(1, 6)]


# === GENE GENERATION ===
def gene_a(nodes, connectivity):
    # generate random cell matrix with $nodes nodes
    # connects node_i to node_j with probability $sparsity thresh
    nn = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in range(nodes):
            if j > i:
                num = np.random.randint(1, len(idx_to_op)) if np.random.rand() < connectivity else 0
                nn[i, j] = num
            if j > 0 and i == nodes - 1 and all(nn[:, j] == 0):
                nn[0, j] = 1
        if i < nodes - 1 and all(nn[i,] == 0):
            nn[i, -1] = 1

    # ensure input cell sends n**2 edges
    if len(nn[0, :].nonzero()[0]) not in powers:
        not_sole_parent = [j for j in range(nodes) if nn[0, j] and len(nn[:, j].nonzero()[0]) > 1]
        fixable = [x for x in range(1, len(not_sole_parent) + 1) if len(nn[0, :].nonzero()[0]) - x in powers]
        if fixable:
            correction = np.random.choice(not_sole_parent, fixable[0], replace=False)
            nn[0, correction] = 0
        else:
            return gene_a(nodes, connectivity)
    return nn


def gene_bc(cell, prob=.5):
    new_cell = cell.copy()
    if random.random() < prob:
        new_cell[-1, -1] = 1
        cnx_idx = cell[0, 1:].nonzero()[0]
        new_cell[1:, 0][cnx_idx] = 1
    return new_cell


# === CELL VALIDITY CHECKING ===
def matching_leaf(value, l):
    return [x for x in l if x[-1]['t'] == value]


def not_matching_path(o, t, l):
    return [x for x in l if x[0]['o'] != o or x[-1]['t'] != t]


def list_flatten(l):
    return [item for sublist in l for item in sublist]


def cell_traversals(cell, verbose=False):
    last_node = cell.shape[0] - 1
    nodes = dict((x, 1) if x == 0 else (x, 0) for x in range(last_node + 1))
    paths = []
    weights = np.zeros(cell.shape)
    matched = True

    for origin in range(last_node):
        for target in range(last_node + 1):
            if target > origin and cell[origin, target]:
                # compute kernels along edge
                edge_frac = 1 / max(1, sum(cell[origin + 1:, origin])) if cell[target, origin] else 1
                edge_weight = nodes[origin] * edge_frac

                if cell[origin, target] == 2:
                    edge_weight *= 2
                elif cell[origin, target] == 3:
                    edge_weight *= .5
                if cell[target, target] == 0 and nodes[target] != 0 and edge_weight != nodes[target]:
                    if verbose:
                        print("Mismatched weights from {}({}) to {}({})".format(origin, edge_weight, target,
                                                                                nodes[target]))
                    matched = False

                weights[origin, target] = edge_weight
                nodes[target] = nodes[target] + edge_weight if cell[target, target] else edge_weight

                # add edge along to corresponding path(s)
                if not matching_leaf(origin, paths):
                    paths.append([{'o': origin, 't': target}])
                else:
                    for path in matching_leaf(origin, paths):
                        paths.append(path + [{'o': origin, 't': target}])

    path_completion = []
    if len([subpath for path in paths for subpath in path]) > 10000:
        passing = False
    else:
        for path in paths:
            for subpath in path:
                if subpath['t'] is last_node:
                    path_completion.append(True)
                else:
                    used_edge = any(
                        [(subpath in path2 or subpath['t'] is last_node) for path2 in paths if path2 is not path])
                    path_completion.append(used_edge)
        all_paths_complete = all(path_completion)
        weight_conserved = nodes[last_node] == 1
        passing = all_paths_complete and weight_conserved and matched

    if verbose:
        print("All Paths: {}, Node Weights: {}, Matched: {} | PASS: {}".format(all_paths_complete, nodes, matched,
                                                                               passing))
    return passing, weights, matched


# === GENE MODDING ===
def find_partition(l):
    xs = [[] for _ in range(int(sum(l)))]
    for n in sorted(l, reverse=True):
        emptiest = np.argmin([sum(x) for x in xs])
        if sum(xs[emptiest]) < 1:
            xs[emptiest].append(n)
    return xs


def add_node(cell, weights):
    # find number of additional output nodes needed
    out = cell
    partitions = find_partition(weights)
    new_nodes = len([x for x in partitions if x != [1]])
    mod_index = cell.shape[0] - 1
    target_index = mod_index
    if new_nodes:
        out = np.pad(out, ((0, new_nodes), (0, new_nodes)), mode='constant', constant_values=0)

    # add nodes for each partition
    for partition in partitions:
        for weight in partition:
            # find first matching edge
            change = [i for (i, x) in enumerate(weights) if x == weight][0]

            if weight == 1:
                # link correct weighted edges to output
                out[change, -1] = out[change, mod_index]
                out[change, mod_index] = 0
            else:
                # link partitioned edges to intermediate nodes
                out[change, target_index] = out[change, mod_index]
                if target_index != mod_index:
                    out[change, mod_index] = 0
                out[target_index, target_index] = 1
                out[target_index, -1] = 1
                weights[change] = 0

        if partition != [1]:
            target_index += 1
    return out


def gene_mod(cell, verbose=False):
    valid, weights, amends = cell_traversals(cell, verbose)
    new_cell = cell.copy()
    if valid:
        return new_cell

    # delete channel mods if any are present to attempt to remedy sizing issues
    np.place(new_cell, new_cell == 3, 1)
    np.place(new_cell, new_cell == 2, 1)
    valid, weights, amends = cell_traversals(new_cell, verbose)
    if valid:
        return new_cell

    # add intermediate concat nodes to ensure input/output consistency
    out_weight = weights[:, -1]
    if sum(out_weight) % 1 == 0 and max(out_weight) <= 1 and any(x != 1 for x in out_weight):
        new_cell = add_node(new_cell, out_weight)
        valid, weights, amends = cell_traversals(new_cell, verbose)
        if valid:
            return new_cell

    # trial intermediate concat nodes
    for i in range(1, new_cell.shape[0]):
        new_cell_node = new_cell.copy()
        new_cell_node[i, i] = 1
        valid, weights, amends = cell_traversals(new_cell_node, verbose)
        if valid:
            return new_cell_node
    return None


# === GENERATE CELL FROM GENES ===
def gen_cell(nodes, connectivity, concat, verbose=False):
    cell_mod = None
    i=0
    while cell_mod is None:
        if verbose:
            print("=== Building Cell ===")
        cell_a = gene_a(nodes, connectivity)
        cell = gene_bc(cell_a, concat)
        cell_mod = gene_mod(cell, verbose)
        i+=1
    return cell_mod


# === CELL MUTATION ===
def mutate_cell(cell, mutation_probability):
    mutated_cell = cell.copy()
    changes = []
    nodes = mutated_cell.shape[0]
    for i in range(nodes):
        for j in range(nodes):
            if j > i:
                if not mutated_cell[i][j] and np.random.rand() < mutation_probability:
                    new_op = np.random.randint(1, len(idx_to_op))
                    changes.append([i, j, mutated_cell[i, j], new_op, "Insertion"])
                    mutated_cell[i, j] = new_op
                    mutated_cell[i, j] = 1 if mutated_cell[-1, -1] == 1 else 0
                elif mutated_cell[i][j]:
                    if np.random.rand() < mutation_probability:
                        if np.random.rand() > .5:
                            changes.append([i, j, mutated_cell[i, j], 0, "Deletion"])
                            mutated_cell[i, j] = 0
                            mutated_cell[j, i] = 0
                        else:
                            new_op = np.random.randint(1, len(idx_to_op))
                            changes.append([i, j, mutated_cell[i, j], new_op, "Alteration"])
                            mutated_cell[i, j] = new_op

            if j > 0 and i == nodes - 1 and all(mutated_cell[:, j] == 0):
                changes.append([0, j, mutated_cell[i, j], 1, "Restorative"])
                mutated_cell[0, j] = 1
        if i < nodes - 1 and all(mutated_cell[i,] == 0):
            changes.append([i, -1, mutated_cell[i, j], 1, "Restorative"])
            mutated_cell[i, -1] = 1

    change_log = []
    for i, j, old_op, new_op, change_type in changes:
        change_log.append("{}: {}->{} at {},{}".format(change_type,
                                                       idx_to_op[old_op],
                                                       idx_to_op[new_op],
                                                       i, j))
    return mutated_cell


def concat_swap(cell, prob):
    new_cell = cell.copy()
    if random.random() < prob:
        if len(new_cell.diagonal().nonzero()[0]):
            return np.triu(new_cell, k=1)
        else:
            cell_mod = None
            attempts = 0
            while cell_mod is None:
                cell_bc = gene_bc(new_cell, 1)
                cell_mod = gene_mod(cell_bc)
                attempts += 1
                if attempts > 10:
                    return cell
            return cell_mod
    else:
        return cell


def swap_mutate(cell, swap_prob, mutate_prob):
    swapped = concat_swap(cell, swap_prob)
    modded = None
    while modded is None:
        mutated = mutate_cell(swapped, mutate_prob)
        modded = gene_mod(mutated)
    return modded


# === CELL STATS ===
def cell_stats(cell):
    nonzeros = 0
    possible_cnx = 0
    nodes = cell.shape[0]
    for i in range(nodes):
        for j in range(nodes):
            if j > i:
                possible_cnx += 1
                if cell[i][j] != 0:
                    nonzeros += 1
    print("Nodes: {}, Connectivity: {}".format(nodes, nonzeros / possible_cnx))


def build_cell(pairs):
    # build cell matrix from connectivity pairs
    # input [(0,1,3),(1,2,4)] means node0->node1 via op 3, node1->node2 via op4, etc

    nodes = np.max([[x[0], x[1]] for x in pairs]) + 1
    matrix = np.zeros([nodes, nodes])
    for origin, target, function in pairs:
        if type(function) == str:
            matrix[origin, target] = op_to_idx[function]
        else:
            matrix[origin, target] = function
    return matrix


def show_cell(cell, name='Cell'):
    # plot cell DAG
    g = nx.DiGraph()
    node_color_map = []
    edge_color_map = []
    for origin in range(cell.shape[0]):
        g.add_node(origin)
        node_color_map.append("b" if cell[origin, origin] else "r")
    for origin, target in zip(*cell.nonzero()):
        if target > origin:
            edge_color_map.append("b" if cell[target, origin] else "r")
            g.add_edge(origin, target, function=idx_to_op[int(cell[origin, target])])
    pos = graphviz_layout(g, prog='dot')

    size = cell.shape[0]
    plt.figure(figsize=(size * 1.5, size * 1.5))
    nx.draw_networkx(g, pos, node_color=node_color_map, edge_color=edge_color_map)
    edge_labels = nx.get_edge_attributes(g, 'function')
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels, arrows=True)
    plt.title(name)
    plt.show()
