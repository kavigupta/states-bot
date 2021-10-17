from itertools import count

import numpy as np


from .solver.initial_solver import initially_solve
from .solver.refinement_solver import improve_until_convergence, ratio
from .solver.solver_configuration import SolverConfiguration
from .costs import PopulationDeviation

from .graph import Graph
from .data import Metadata
from .assignment import Assignment

from permacache import permacache


def sample_guaranteed(data, *, rng_seed, n_states, config=SolverConfiguration()):
    meta = Metadata(n_states, data.pops, data, data.centers)
    rng = np.random.RandomState(rng_seed)
    graph = Graph(data)
    while True:
        assign = sample(
            [PopulationDeviation()],
            graph=graph,
            rng_seed=rng.choice(2 ** 32),
            n_states=n_states,
            config=config,
        )
        if ratio(graph, assign) < config.equality_ratio_limit:
            return Assignment.from_county_to_state(
                data,
                meta,
                assign,
            )


@permacache(
    "statesbot/run_sampler/sample_2",
    key_function=dict(graph=lambda graph: graph.hash),
)
def sample(costs, graph, *, rng_seed, n_states, config):
    rng = np.random.RandomState(rng_seed)

    assign = initially_solve(
        graph,
        np.arange(n_states),
        rng.choice(2 ** 32),
        config=config,
    )

    for i in set(assign):
        assert graph.subset_connected(np.where(assign == i)[0])

    assign = improve_until_convergence(
        costs,
        graph,
        assign,
        random_seed=rng.choice(2 ** 32),
        config=config,
    )

    for i in set(assign):
        assert graph.subset_connected(np.where(assign == i)[0])

    return assign
