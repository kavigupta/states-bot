from collections import defaultdict

import us
import numpy as np
import pandas as pd
import tqdm
import attr

import geonamescache

import electiondata as e
from electiondata.examples.census_2010_pop import Census2010Population
from electiondata.examples.plotly_geojson import PlotlyGeoJSON

from permacache import permacache, stable_hash
from methodtools import lru_cache


class Data:
    version = 1.0

    def __init__(self):
        geojson = PlotlyGeoJSON().get()
        self.feats = [
            f
            for f in geojson["features"]
            if f["id"][:2] not in {"72", "15", "02"}
            and f["id"] not in {"25019", "53055"}
        ]
        self.fipses = [f["id"] for f in self.feats]
        self.areas = np.array([f["properties"]["CENSUSAREA"] for f in self.feats])
        self.original_states = get_states(self.fipses)
        self.geojson = dict(type=geojson["type"], features=self.feats)
        data = Census2010Population(e.alaska.FIPS).get()
        pop_by_fips = dict(zip(data.FIPS, data.CENSUS2010POP))
        self.pops = np.array([pop_by_fips[f] for f in self.fipses])
        self.coords = [get_coords(feat) for feat in self.feats]
        self.centers = np.array([c.mean(0) for c in self.coords])
        self.neighbors = all_edges(self.coords)
        self.weights = weights(self.feats, self.coords, self.neighbors)

        self.cities = cities_dataset(self.fipses)

    def name_state_by_real_states(self, counties):
        contained_states = []
        for state, scounties in self.original_states.items():
            overlap = list(set(counties) & set(scounties))
            overlap = self.pops[overlap].sum()
            if overlap / self.pops[scounties].sum() > 0.8:
                contained_states.append((overlap, state))
        contained_states = sorted(contained_states, reverse=True)
        if not contained_states:
            return None
        if len(contained_states) == 1:
            return contained_states[0][1]
        (_, s1), (_, s2), *_ = contained_states
        return combine_names(s1, s2)

    def name_state(self, counties):
        name = self.name_state_by_real_states(counties)
        if name is not None:
            return name

        cities_for_state = [city for county in counties for city in self.cities[county]]
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


def get_states(fipses):
    by_state = defaultdict(list)
    for i, fips in enumerate(fipses):
        by_state[us.states.lookup(fips[:2]).name].append(i)
    return dict(by_state.items())


@attr.s
class Metadata:
    count = attr.ib()
    stat = attr.ib()
    graph = attr.ib()
    centers = attr.ib()

    def bordering(self, a, b):
        return any(x in a for y in b for x in self.graph.neighbors[y])


def get_coords(feat):
    return np.array([x for x in feat["geometry"]["coordinates"] for x in x for x in x])


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


def weights(feats, coords, neighbors):
    perimiters = {}
    for a in range(len(feats)):
        c = feats[a]["geometry"]["coordinates"]
        c = [np.array(x) for y in c for x in y]
        perimiters[a] = sum([(((x[1:] - x[:-1]) ** 2).sum(-1) ** 0.5).sum() for x in c])

    weight = {}
    for a, n_a in enumerate(neighbors):
        for b in n_a:
            if a == b:
                continue
            in_a, _ = np.where(
                np.abs(coords[a][:, None] - coords[b][None]).sum(-1) < 0.1
            )
            in_a = sorted(set(in_a))
            total = 0
            for start, end in zip(in_a, in_a[1:]):
                if start == end - 1:
                    total += ((coords[a][start] - coords[a][end]) ** 2).sum() ** 0.5
            weight[a, b] = total / perimiters[a]
    return weight


def cities_dataset(fipses):
    cities_dataset = pd.read_csv(
        "https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv",
        sep="|",
    )
    cities_dataset = cities_dataset[cities_dataset.County == cities_dataset.County]
    normalizer = e.usa_county_to_fips("State short")
    normalizer.rewrite["hoonah angoon"] = "hoonah-angoon"
    normalizer.rewrite["matanuska susitna"] = "matanuska-susitna"
    normalizer.rewrite["prince of wales hyder"] = "prince of wales-hyder"
    normalizer.rewrite["valdez cordova"] = "valdez-cordova"
    normalizer.rewrite["yukon koyukuk"] = "yukon-koyukuk"
    normalizer.rewrite["saint louis"] = "saint louis city"
    normalizer.rewrite["northern mariana islands"] = "northern islands"
    normalizer.rewrite["baltimore"] = "baltimore city"
    normalizer.rewrite["saint thomas"] = "saint thomas island"
    normalizer.rewrite["franklin"] = "franklin city"
    normalizer.rewrite["richmond"] = "richmond city"
    normalizer.rewrite["saint croix"] = "saint croix island"
    normalizer.rewrite["bedford"] = "bedford city"
    normalizer.rewrite["fairfax"] = "fairfax city"
    normalizer.rewrite["roanoke"] = "roanoke city"
    normalizer.rewrite["saint john"] = "saint john island"
    normalizer.rewrite["american samoa"] = "ERROR"
    normalizer.rewrite["federated states of micro"] = "ERROR"
    normalizer.rewrite["marshall islands"] = "ERROR"
    normalizer.rewrite["palau"] = "ERROR"

    normalizer.apply_to_df(cities_dataset, "County", "FIPS", var_name="normalizer")
    cd = e.remove_errors(cities_dataset, "FIPS")

    backmap = {fips: i for i, fips in enumerate(fipses)}
    result = {}
    for city, state, fips in zip(cd.City, cd["State short"], cd.FIPS):
        if (city, state) in result:
            continue
        if fips in backmap:
            result[city, state] = backmap[fips]

    usa = [
        city
        for city in geonamescache.GeonamesCache().get_cities().values()
        if city["countrycode"] == "US"
    ]
    cities = [[] for _ in range(len(fipses))]
    for city in usa:
        key = city["name"].replace("St.", "Saint"), city["admin1code"]
        if key not in result:
            continue
        cities[result[key]].append(city)
    return cities
