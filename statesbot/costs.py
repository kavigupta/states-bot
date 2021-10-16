import attr

import numpy as np


@attr.s
class PopulationDeviation:
    def __call__(self, graph, region1, region2):
        a, b = graph.total_eqstat(region1), graph.total_eqstat(region2)
        return np.abs(a - b) / (a + b)
