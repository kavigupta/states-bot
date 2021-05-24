import numpy as np
import tqdm
import attr

import electiondata as e
from electiondata.examples.census_2010_pop import Census2010Population
from electiondata.examples.plotly_geojson import PlotlyGeoJSON

from permacache import permacache, stable_hash
from methodtools import lru_cache


class Data:
    def __init__(self):
        geojson = PlotlyGeoJSON().get()
        self.feats = [
            f for f in geojson["features"] if f["id"][:2] not in {"72", "15", "02"}
        ]
        self.fipses = [f["id"] for f in self.feats]
        self.geojson = dict(type=geojson["type"], features=self.feats)
        data = Census2010Population(e.alaska.FIPS).get()
        pop_by_fips = dict(zip(data.FIPS, data.CENSUS2010POP))
        self.pops = np.array([pop_by_fips[f] for f in self.fipses])
        self.coords = [get_coords(feat) for feat in self.feats]
        self.centers = np.array([c.mean(0) for c in self.coords])
        self.neighbors = all_edges(self.coords)

    @lru_cache(None)
    def kaway(self, x, k):
        if k == 0:
            return {x}
        return {z for y in self.kaway(x, k - 1) for z in self.neighbors[y]}


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
