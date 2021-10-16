from collections import defaultdict

import numpy as np
import tqdm
import attr

from permacache import permacache, stable_hash
from methodtools import lru_cache

from statesbot.naming import name_state

from .counties import get_counties
from .countylikes import get_countylikes

SPECIAL_EDGES = [
    ("51099", "24017"),  # south of dc
    ("36103", "09001"),  # long island connected to conneticut
    ("36103", "09009"),
    ("36103", "09007"),
    ("36103", "09011"),
    ("26097", "26047"),  # michigan's parts connected
    ("51131", "51810"),  # virginia beach
    ("25019", "25007"),  # Nantucket Island
    ("25019", "25001"),
    ("53055", "53057"),  # san juan island
    ("53055", "53073"),
    ("06073", "15001"),  # San Diego --> Hawaii
    ("15001", "15009"),  # Hawaii --> Maui
    ("15009", "15003"),  # Maui --> Honolulu
    ("15003", "15007"),  # Honolulu --> Kauai
    ("53073", "02AL"),  # Whatcom --> Alaska
    ("12086.Homestead", "72005"),  # Miami-Dade --> Auguadilla, PR
    ("72147", "72037"),  # Vieques island to Ceiba
    ("72049", "72037"),  # Culebra island to Ceiba
]


ISLANDS = {"02AL"}


class Data:
    version = 1.13

    def __init__(self, use_countylikes=True):
        self.countylikes = get_countylikes() if use_countylikes else get_counties()
        self.ident_back = {x.ident: i for i, x in enumerate(self.countylikes)}
        self.pops = np.array([f.pop for f in self.countylikes])

        self.original_states = get_states(self.countylikes)
        self.centers = np.array([c.center for c in self.countylikes])
        self.neighbors = all_edges(self.countylikes)
        for x, y in SPECIAL_EDGES:
            x, y = self.ident_back[x], self.ident_back[y]
            self.neighbors[x].add(y)
            self.neighbors[y].add(x)
        self.weights = weights(self.countylikes, self.neighbors)

    @property
    def geojson(self):
        return dict(
            type="FeatureCollection", features=[f.feature for f in self.countylikes]
        )

    def name_state(self, counties):
        return name_state(
            get_states(self.countylikes),
            self.pops,
            [x.cities for x in self.countylikes],
            counties,
        )


def get_states(countylikes):
    by_state = defaultdict(list)
    for i, c in enumerate(countylikes):
        by_state[c.state].append(i)
    return dict(by_state.items())


@attr.s
class Metadata:
    count = attr.ib()
    stat = attr.ib()
    graph = attr.ib()
    centers = attr.ib()

    def bordering(self, a, b):
        return any(x in a for y in b for x in self.graph.neighbors[y])


@permacache("statesbot/data/all_edges_2", key_function=dict(all_edges=stable_hash))
def all_edges(countylikes, **kwargs):
    return [edges(countylikes, i, **kwargs) for i in tqdm.trange(len(countylikes))]


def edges(countylikes, i, centroid_distance=5, actual_distance=1e-2):
    if countylikes[i].ident in ISLANDS:
        return set()
    results = set()
    for j, c in enumerate(countylikes):
        if c.ident in ISLANDS:
            continue
        centroid_distance_this = np.abs(
            c.flat_coords.mean(0) - countylikes[i].flat_coords.mean(0)
        ).sum()
        if centroid_distance_this > centroid_distance:
            continue
        distance = np.partition(
            np.abs(countylikes[i].flat_coords[:, None] - c.flat_coords[None])
            .sum(-1)
            .flatten(),
            2,
        )[2]
        if distance <= actual_distance:
            results.add(j)
    return results


def weights(countylikes, neighbors):
    weight = {}
    for a, n_a in enumerate(neighbors):
        for b in n_a:
            if a == b:
                continue
            in_a, _ = np.where(
                np.abs(
                    countylikes[a].flat_coords[:, None]
                    - countylikes[b].flat_coords[None]
                ).sum(-1)
                < 0.1
            )
            in_a = sorted(set(in_a))
            total = 0
            for start, end in zip(in_a, in_a[1:]):
                if start == end - 1:
                    total += (
                        (
                            countylikes[a].flat_coords[start]
                            - countylikes[a].flat_coords[end]
                        )
                        ** 2
                    ).sum() ** 0.5
            weight[a, b] = total
    return weight
