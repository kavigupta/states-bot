from itertools import count

import numpy as np
from .data import Metadata
from .states_sampler import sample_states
from .assignment import Assignment

from permacache import permacache


@permacache(
    "statesbot/run_sampler/sample",
    key_function=dict(data=lambda data: data.version, pbar=None),
)
def sample(data, *, rng_seed, n_states, pbar):
    meta = Metadata(n_states, data.pops, data, data.centers)
    rng = np.random.RandomState(rng_seed)
    best = sample_initial(meta, rng)
    assign = Assignment.from_county_to_state(
        meta, best.county_to_state_no_nan.astype(int)
    )
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


def sample_initial(meta, rng):
    while True:
        result = sample_states(meta, rng.randint(2 ** 32))
        if not np.isnan(result.county_to_state).any():
            return result
