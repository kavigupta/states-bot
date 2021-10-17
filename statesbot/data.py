from collections import defaultdict

import numpy as np
from shapely import geometry
from shapely.ops import transform
import tqdm
import attr

from statesbot.naming import name_state

from .counties import Countylike

from .geographies.usa import usa_subcounties_dataset
from .geographies.cities import classify_cities
from .geometry.construct_graph import get_edges

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
    ("5303393616", "53053"),  # vashon island
    ("5303393616", "53035"),
    ("5303393616", "5303391140"),
    ("5303393616", "5303392928"),
    ("0607392780", "15001"),  # San Diego --> Hawaii
    ("15001", "15009"),  # Hawaii --> Maui
    ("15009", "15003"),  # Maui --> Honolulu
    ("15003", "15007"),  # Honolulu --> Kauai
    ("53073", "02198"),  # Whatcom --> Alaska
    ("12087", "72005"),  # Miami-Dade --> Auguadilla, PR
    ("72147", "72037"),  # Vieques island to Ceiba
    ("72049", "72037"),  # Culebra island to Ceiba
    ("02016", "02013"),  # Aluetians West -> Alutians East
]


def to_spherical(lon, lat):
    lon, lat = lon / 180 * np.pi, lat / 180 * np.pi
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z]).T


class Data:
    version = 1.13

    def __init__(self):
        self.table = usa_subcounties_dataset().copy()
        self.table["spherical_geometry"] = self.table.geometry.apply(
            lambda x: transform(to_spherical, x)
        )
        self.ident_back = {x: i for i, x in enumerate(self.table.ident)}
        self.cities = {
            self.ident_back[k]: v for k, v in classify_cities(self.table).items()
        }
        self.countylikes = [
            Countylike(
                row.ident,
                self.cities[self.ident_back[row.ident]],
                row.geometry,
                row.population,
                row.dem_2020,
            )
            for (_, row) in self.table.iterrows()
        ]
        self.pops = np.array([f.pop for f in self.countylikes])

        self.centers = np.array([c.center for c in self.countylikes])

        self.weights = dict(get_edges(self.table).items())
        self.weights.update({(v, u): w for (u, v), w in self.weights.items()})

        self.neighbors = [set() for _ in range(len(self.table))]
        for x, y in self.weights:
            self.neighbors[x].add(y)
            self.neighbors[y].add(x)
        for x, y in SPECIAL_EDGES:
            x, y = self.ident_back[x], self.ident_back[y]
            assert x not in self.neighbors[y], str((x, y))
            self.neighbors[x].add(y)
            self.neighbors[y].add(x)
            self.weights[x, y] = self.weights[y, x] = 0

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

    @property
    def idents(self):
        return [c.ident for c in self.countylikes]

    def polygon(self, county):
        return self.countylikes[county].polygon


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
