from itertools import count

import numpy as np
from .data import Metadata
from .states_sampler import sample_states
from .assignment import Assignment

from permacache import permacache


def sample_guaranteed(data, *, rng_seed, n_states, pbar, bar=2.5):
    meta = Metadata(n_states, data.pops, data, data.centers)
    rng = np.random.RandomState(rng_seed)
    while True:
        assign = Assignment.from_county_to_state(
            data,
            meta,
            sample(data, rng_seed=rng.choice(2 ** 32), n_states=n_states, pbar=pbar, filter_bar=bar*2),
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
    assign = sample_initial(data, meta, rng, filter_bar=filter_bar)

    for idx in pbar(count()):
        if not assign.fix_border():
            break
        if idx % 10 == 0:
            for state_idx in range(assign.meta.count):
                while assign.remove_farthest(state_idx, rng):
                    pass
        frac = assign.aggregated_stats.max() / assign.aggregated_stats.min()
        if idx % 100 == 0:
            print(idx, frac)
    return assign.county_to_state


def sample_initial(data, meta, rng, filter_bar):
    while True:
        result = sample_states(meta, rng.randint(2 ** 32))
        if np.isnan(result.county_to_state).any():
            continue
        print("VALID")
        assign = Assignment.from_county_to_state(
            data, meta, result.county_to_state_no_nan.astype(int)
        )
        if assign.aggregated_stats.max() / assign.aggregated_stats.min() > filter_bar:
            continue
        return assign
