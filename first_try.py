import attr
from collections import defaultdict


@attr.s
class Assignment:
    state_to_counties = attr.ib()
    county_to_state = attr.ib()

    def assign(self, county, state):
        current_state = self.county_to_state[county]
        self.county_to_state[county] = state
        self.state_to_counties[current_state].remove(county)
        self.state_to_counties[state].add(county)

    @property
    def state_lists(self):
        return [list(x) for _, x in sorted(self.state_to_counties.items())]

    @property
    def centroids(self):
        return np.array([(centers[x] * pops[x][:,None]).sum(0) / pops[x].sum(0) for x in self.state_lists])

    @property
    def populations(self):
        return np.array([pops[x].sum() / 1e6 for x in self.state_lists])

    @classmethod
    def from_county_to_state(cls, county_to_state):
        state_to_counties = defaultdict(set)
        for county, state in enumerate(county_to_state):
            state_to_counties[state].add(county)
        return cls(state_to_counties, county_to_state)

    @classmethod
    def random_init(cls, count, rng):
        p = pops / pops.sum()
        center_idxs = rng.choice(centers.shape[0], size=count, p=p, replace=False)
        return cls.assign_by_distance(centers[center_idxs])

    @classmethod
    def assign_by_distance(cls, center_pos, weights=1):
        county_to_state = (
            distances(center_pos[None], centers[:, None]) * weights
        ).argmin(1)
        return cls.from_county_to_state(county_to_state)

    def neighboring_states(self, county):
        return {self.county_to_state[c] for c in all_edges[county]}

    def rearrange(self, rng, weight_centerness):
        centroids = self.centroids
        populs = self.populations
        populs_mean = populs.mean()

        count = 0
        counties = list(range(len(self.county_to_state)))
        rng.shuffle(counties)
        for county in counties:
            current_state = self.county_to_state[county]
            best_state = None
            best_score = 0
            for other_state in self.neighboring_states(county):
                if current_state == other_state:
                    continue
                new_populs = populs.copy()
                new_populs[other_state] += pops[county] / 1e6
                new_populs[current_state] -= pops[county] / 1e6
                dpop = (
                    (np.abs(new_populs - populs_mean) ** 2).sum() ** 0.5
                    - (np.abs(populs - populs_mean) ** 2).sum() ** 0.5
                )
                d1, d2 = distances(
                    np.array([[centers[county]]]),
                    np.array([centroids[[current_state, other_state]]]),
                )[0]
                dcenter = d2 - d1
                dcenter *= weight_centerness
                if dpop + dcenter < best_score:
                    best_score = dpop + dcenter
                    best_state = other_state
            if best_state is not None:
                self.assign(county, best_state)
                count += 1
                if count >= 100:
                    break
        return count

rng = np.random.RandomState(2)
assign = Assignment.random_init(count, rng)
assign = assign.assign_by_distance(assign.centroids)
for idx in tqdm.trange(1000):
    z = assign.rearrange(rng, 4)
    if z < 10:
        break
    if idx % 25 == 0:
        print(idx, z, [np.percentile(assign.populations, k) for k in (0, 25, 50, 75, 100)])
