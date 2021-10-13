import attr
import numpy as np

import networkx as nx


@attr.s
class States:
    meta = attr.ib()
    target = attr.ib()
    states = attr.ib()
    state_pops = attr.ib()
    state_neighbor_counties = attr.ib()
    used_counties = attr.ib()
    given_up = attr.ib()
    county_to_state = attr.ib()

    @property
    def county_to_state_no_nan(self):
        return np.nan_to_num(self.county_to_state)

    @staticmethod
    def setup(meta):
        return States(
            meta=meta,
            target=sum(meta.stat) / meta.count,
            states=[set() for _ in range(meta.count)],
            state_pops=[0 for _ in range(meta.count)],
            state_neighbor_counties=[set() for _ in range(meta.count)],
            used_counties=set(),
            given_up=set(),
            county_to_state=np.zeros(len(meta.stat)) + np.nan,
        )

    def assign(self, state, county):
        assert county not in self.used_counties
        self.used_counties.add(county)
        for s in self.state_neighbor_counties:
            if county in s:
                s.remove(county)
        self.states[state].add(county)
        self.state_pops[state] += self.meta.stat[county]
        self.state_neighbor_counties[state].update(
            x for x in self.meta.graph.neighbors[county] if x not in self.used_counties
        )
        self.county_to_state[county] = state

    def initial_assign(self, rng):
        [linchpins] = np.where(
            self.meta.stat > self.meta.stat.sum() / len(self.states) / 2
        )
        for state, county in enumerate(linchpins[: len(self.states)]):
            self.assign(state, county)
        others = list(set(range(len(self.meta.stat))) - set(linchpins))
        rng.shuffle(others)
        for state in range(len(linchpins), len(self.states)):
            self.assign(state, others.pop())

    def smallest_state(self):
        not_given_up = [
            idx for idx in range(len(self.states)) if not idx in self.given_up
        ]
        if not not_given_up:
            return None
        return min(
            not_given_up,
            key=lambda i: (
                self.state_pops[i] >= self.target,
                max(len(self.valid_neighbors(i)), 20) / 20
                + 2 * self.state_pops[i] / self.target,
            ),
        )

    def valid_neighbors(self, state):
        return sorted(
            x
            for x in self.state_neighbor_counties[state]
            if self.meta.stat[x] + self.state_pops[state] <= self.target * 1.5
        )

    def assign_to_smallest(self):
        smallest = self.smallest_state()
        if smallest is None:
            return True
        valid_neighbors = self.valid_neighbors(smallest)
        if not valid_neighbors:
            if not self.state_neighbor_counties[smallest]:
                if self.state_pops[smallest] / self.target < 1 / 6:
                    return True  # fail fast
                self.given_up.add(smallest)
                return
            valid_neighbors = self.state_neighbor_counties[smallest]

        self.assign(
            smallest, min(valid_neighbors, key=lambda i: self.closeness(i, smallest))
        )
        return False

    def closeness(self, county, state):
        window = list(self.meta.graph.kaway(county, 3))
        by_state = self.county_to_state[window]
        near_this = (by_state == state).sum()
        near_other = len(
            [x for x in by_state if x != x or self.state_pops[int(x)] < self.target]
        )
        return near_other - near_this


def sample_states(meta, seed):
    states = States.setup(meta)
    rng = np.random.RandomState(seed)
    states.initial_assign(rng)
    while True:
        if states.assign_to_smallest():
            break
    print(np.max(states.state_pops) / np.min(states.state_pops))
    return states
