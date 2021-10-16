from itertools import count

import numpy as np

from .graph import Graph
from .data import Metadata
from .assignment import Assignment

from .solver.initial_solver import initially_solve

from permacache import permacache


def sample_guaranteed(data, *, rng_seed, n_states, pbar, bar=1.8):
    meta = Metadata(n_states, data.pops, data, data.centers)
    rng = np.random.RandomState(rng_seed)
    while True:
        assign = Assignment.from_county_to_state(
            data,
            meta,
            sample(
                data,
                rng_seed=rng.choice(2 ** 32),
                n_states=n_states,
                pbar=pbar,
                filter_bar=(bar - 1) * 2 + 1,
            ),
        )
        frac = assign.aggregated_stats.max() / assign.aggregated_stats.min()
        if frac < bar:
            return assign


@permacache(
    "statesbot/run_sampler/sample",
    key_function=dict(data=lambda data: data.version, pbar=None),
)
def sample(data, *, rng_seed, n_states, pbar, filter_bar):
    meta = Metadata(n_states, data.pops, data, data.centers)
    rng = np.random.RandomState(rng_seed)

    assign = initially_solve(
        Graph(data), np.arange(n_states), rng.choice(2 ** 32), filter_bar
    )
    assign = Assignment.from_county_to_state(data, meta, assign.astype(int))

    for idx in pbar(range(max(1000, n_states * 40))):
        if not assign.fix_border(idx):
            break
        if idx % 10 == 0:
            for state_idx in range(assign.meta.count):
                while assign.remove_farthest(state_idx, rng):
                    pass
        frac = assign.aggregated_stats.max() / assign.aggregated_stats.min()
        if idx % 100 == 0:
            print(idx, frac)
    return assign.county_to_state
