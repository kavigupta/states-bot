from collections import defaultdict
import networkx as nx
import tqdm.auto as tqdm
import numpy as np

from .error_code import SolverFailure


def region_graph(graph, assign):
    result = nx.Graph()
    result.add_nodes_from(set(assign))
    for u, v in graph.edges:
        u, v = assign[u], assign[v]
        if u == v:
            continue
        result.add_edge(u, v)
    return result


def compute_cost(costs, graph, assign, l1, l2):
    return sum(
        cost(graph, np.where(assign == l1)[0], np.where(assign == l2)[0])
        for cost in costs
    )


def ratio(graph, assign):
    regions = set(assign)
    areas = [graph.total_eqstat(np.where(assign == r)) for r in regions]
    return max(areas) / min(areas)


def improve_until_convergence(costs, graph, assign, random_seed, *, config):
    assign = assign.copy()
    rng = np.random.RandomState(random_seed)
    edge_retry_counts = defaultdict(int)
    pbar = tqdm.tqdm()
    while True:
        rg = region_graph(graph, assign)
        problematic_edges = sorted(
            rg.edges,
            key=lambda uv: compute_cost(costs, graph, assign, *uv),
            reverse=True,
        )[: int(len(rg.edges) * config.improvement_fraction_edges)]

        pbar.set_description(f"ratio={ratio(graph, assign):.2f}")
        pbar.total = len(problematic_edges) * config.improvement_retries_per_edge
        pbar.n = pbar.last_print_n = sum(
            edge_retry_counts[edge] for edge in problematic_edges
        )
        pbar.refresh()
        valid_edges = [
            edge
            for edge in problematic_edges
            if edge_retry_counts[edge] < config.improvement_retries_per_edge
        ]
        if not valid_edges:
            pbar.close()
            break
        for edge in valid_edges:
            if not attempt_improvement(costs, graph, assign, rng, *edge):
                pbar.update()
                edge_retry_counts[edge] += 1
    print(ratio(graph, assign))
    return assign


def attempt_improvement(costs, graph, assign, rng, first, second):
    [combo_arr] = np.where((assign == first) | (assign == second))
    g = combination_graph(graph, combo_arr)
    direction = rng.randn(graph.euclidean.shape[-1])
    along_axis = graph.euclidean[combo_arr] @ direction
    source, target = combo_arr[[along_axis.argmin(), along_axis.argmax()]]
    try:
        left, right = balanced_min_cut(graph, g, source, target)
    except SolverFailure:
        return False
    initial_cost = compute_cost(costs, graph, assign, first, second)
    new_cost = sum(cost(graph, left, right) for cost in costs)
    if new_cost > initial_cost:
        return False
    assign[left] = first
    assign[right] = second
    return True


def combination_graph(graph, combo_arr):
    combo = set(combo_arr.tolist())
    g = nx.Graph()
    for c in combo:
        g.add_node(c, merged_nodes=[c])
    for u in combo:
        for v in graph.neighbors(u):
            if u != v and v in combo:
                g.add_edge(u, v, weight=graph.weight(u, v))
    return g


def balanced_min_cut(original_graph, g, source, target):
    if len(g.nodes) == 2:
        return g.nodes[source]["merged_nodes"], g.nodes[target]["merged_nodes"]

    nodes = [source, target]

    _, sides = nx.algorithms.flow.minimum_cut(g, source, target, capacity="weight")

    eqstats = [
        original_graph.total_eqstat(
            [x for node in side for x in g.nodes[node]["merged_nodes"]]
        )
        for side in sides
    ]

    index = np.argmin(eqstats)
    if len(sides[index]) > 1:
        g, nodes[index] = merge_nodes_in(g, sides[index])
    else:
        candidates = [x for x in g.neighbors(nodes[index]) if x != nodes[1 - index]]
        if not candidates:
            raise SolverFailure
        other = max(
            candidates,
            key=lambda x: g.edges[nodes[index], x]["weight"],
        )
        g, nodes[index] = merge_nodes_in(g, {nodes[index], other})
    return balanced_min_cut(original_graph, g, *nodes)


def merge_nodes_in(g, nodes):
    total_weight = defaultdict(float)
    for u in nodes:
        for v in g.neighbors(u):
            if v in nodes:
                continue
            total_weight[v] += g.edges[u, v]["weight"]
    first, *rest = list(nodes)
    full_merged = [x for node in nodes for x in g.nodes[node]["merged_nodes"]]
    for r in rest:
        g.remove_node(r)
    g.nodes[first]["merged_nodes"] = full_merged
    for v, w in total_weight.items():
        g.add_edge(first, v, weight=w)
    return g, first
