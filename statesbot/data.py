from collections import defaultdict

import numpy as np
import tqdm
import attr

from permacache import permacache, stable_hash
from methodtools import lru_cache

from .countylikes import get_countylikes


class Data:
    version = 1.0

    def __init__(self):
        self.countylikes = get_countylikes()
        self.pops = np.array([f.pop for f in self.countylikes])

        self.original_states = get_states(self.countylikes)
        self.centers = np.array([c.center for c in self.countylikes])
        self.neighbors = all_edges([x.flat_coords for x in self.countylikes])
        self.weights = weights(self.countylikes, self.neighbors)

    @property
    def geojson(self):
        return dict(
            type="FeatureCollection", features=[f.feature for f in self.countylikes]
        )

    def name_state_by_real_states(self, counties):
        contained_states = []
        for state, scounties in self.original_states.items():
            overlap = list(set(counties) & set(scounties))
            overlap = self.pops[overlap].sum()
            if overlap / self.pops[scounties].sum() > 2 / 3:
                contained_states.append((overlap, state))
        contained_states = sorted(contained_states, reverse=True)
        if (
            sum(c[0] for c in contained_states[:2])
            < self.pops[list(counties)].sum() / 2
        ):
            return None
        if len(contained_states) == 1:
            return contained_states[0][1]
        (_, s1), (_, s2), *_ = contained_states
        return combine_names(s1, s2)

    def name_state(self, counties):
        name = self.name_state_by_real_states(counties)
        if name is not None:
            return name

        cities_for_state = [
            city for county in counties for city in self.countylikes[county].cities
        ]
        biggest_city = max(cities_for_state, key=lambda x: x["population"])
        if len(counties) == 1:
            return biggest_city["name"]
        cities_for_state = sorted(
            [x for x in cities_for_state if x != biggest_city],
            key=lambda x: x["population"],
        )[-5:]
        distances = [
            (x["latitude"] - biggest_city["latitude"]) ** 2
            + (x["longitude"] - biggest_city["longitude"]) ** 2
            for x in cities_for_state
        ]

        distances = np.array(distances) / max(distances)
        next_city = max(
            zip(cities_for_state, distances),
            key=lambda x: x[0]["population"] / biggest_city["population"] + x[1],
        )[0]
        return combine_names(biggest_city["name"], next_city["name"])

    @lru_cache(None)
    def kaway(self, x, k):
        if k == 0:
            return {x}
        return {z for y in self.kaway(x, k - 1) for z in self.neighbors[y]}


def combine_names(a, b):
    common_letters = set(a[len(a) // 2 :]) & set(b[: len(b) // 2])
    if not common_letters:
        while not b[0].lower() in "aeiou":
            b = b[1:]
        b = b[1:]
        while not a[-1].lower() in "aeiou":
            a = a[:-1]
        return a + b
    best_indices = float("inf"), float("inf")
    for c in common_letters:
        idx = a[::-1].index(c), b.index(c)
        if sum(idx) < sum(best_indices):
            best_indices = idx
    return a[: -best_indices[0] - 1] + b[best_indices[1] :]


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


@permacache("statesbot/data/all_edges", key_function=dict(all_edges=stable_hash))
def all_edges(coords, **kwargs):
    return [edges(coords, i, **kwargs) for i in tqdm.trange(len(coords))]


def edges(coords, i, centroid_distance=5, actual_distance=1e-2):
    distances = np.array(
        [
            np.partition(np.abs(coords[i][:, None] - c[None]).sum(-1).flatten(), 1)[1]
            if (np.abs(c.mean(0) - coords[i].mean(0)) < centroid_distance).all()
            else float("inf")
            for c in coords
        ]
    )
    [within_eps] = np.where(distances <= actual_distance)
    return set(within_eps)


def weights(countylikes, neighbors):
    perimiters = {}
    for a in range(len(countylikes)):
        c = countylikes[a].coordinates
        c = [np.array(x) for y in c for x in y]
        perimiters[a] = sum([(((x[1:] - x[:-1]) ** 2).sum(-1) ** 0.5).sum() for x in c])

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
            weight[a, b] = total / perimiters[a]
    return weight
