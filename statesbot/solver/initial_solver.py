import numpy as np

from .error_code import SolverFailure


def initially_solve(graph, labels, rng_seed, filter_bar, **kwargs):
    rng = np.random.RandomState(rng_seed)
    while True:
        try:
            indices, out = _initially_solve(
                graph, labels, rng.choice(2 ** 32), graph.vertex_indices, **kwargs
            )
            result = np.empty(indices.size, dtype=out.dtype)
            result[indices] = out
        except SolverFailure:
            continue
        each = [graph.total_eqstat(np.where(result == label)[0]) for label in labels]
        ratio = max(each) / min(each)
        print(ratio)
        if ratio < filter_bar:
            return result


def _initially_solve(graph, labels, rng_seed, indices, direction_choices=5):
    indices = np.array(indices)
    if len(labels) > len(indices):
        raise SolverFailure
    if len(labels) == 1:
        return indices, np.array([labels[0]] * len(indices))

    split = len(labels) // 2

    rng = np.random.RandomState(rng_seed)
    direction = rng.randn(graph.euclidean.shape[-1], direction_choices)
    in_direction = graph.euclidean[indices] @ direction
    true_distances = (
        (
            graph.euclidean[indices[in_direction.argmax(0)]]
            - graph.euclidean[indices[in_direction.argmin(0)]]
        )
        ** 2
    ).sum(-1)
    in_direction = in_direction[:, true_distances.argmax()]
    indices = indices[np.argsort(in_direction)]

    eqstat = graph.eqstat[indices]
    eqstat_target = split / len(labels) * eqstat.sum()

    eqstat_cs = np.cumsum(eqstat)

    x = np.searchsorted(eqstat_cs, eqstat_target)

    if x == 0 or x == len(indices):
        raise SolverFailure

    indices_left, indices_right = indices[:x], indices[x:]

    indices_left, indices_right = rectify_connectedness(
        graph, indices_left, indices_right
    )

    real_sum_left = graph.total_eqstat(indices_left)
    split = int(
        np.clip(
            np.round(real_sum_left / eqstat.sum() * len(labels)), 1, len(labels) - 1
        )
    )

    labels_left, labels_right = labels[:split], labels[split:]

    indices_left, out_left = _initially_solve(
        graph, labels_left, rng.choice(2 ** 32), indices_left
    )
    indices_right, out_right = _initially_solve(
        graph, labels_right, rng.choice(2 ** 32), indices_right
    )

    return np.concatenate([indices_left, indices_right]), np.concatenate(
        [out_left, out_right]
    )


def rectify_connectedness(graph, s, t):
    s, t = list(s), list(t)
    s, extra = nonconnected_parts(graph, s)
    t += extra
    t, extra = nonconnected_parts(graph, t)
    s += extra
    if not graph.subset_connected(s):
        raise SolverFailure
    return s, t


def nonconnected_parts(graph, s):
    components_s = [list(x) for x in graph.connected_components(s)]
    if len(components_s) == 1:
        return components_s[0], []
    best_idx = max(
        range(len(components_s)), key=lambda idx: graph.total_eqstat(components_s[idx])
    )
    return components_s[best_idx], [
        x for i, xs in enumerate(components_s) if i != best_idx for x in xs
    ]
