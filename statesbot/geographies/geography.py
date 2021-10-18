from abc import ABC, abstractmethod

import attr
import numpy as np
from permacache import permacache

from shapely.ops import transform

from statesbot.geometry.construct_graph import get_edges
from shapely import geometry

from statesbot.naming import name_state

from .cities import classify_cities


@attr.s
class Geography:
    table = attr.ib()
    cities = attr.ib()
    centers = attr.ib()
    weights = attr.ib()
    neighbors = attr.ib()
    regions = attr.ib()
    max_states = attr.ib()
    atlas_types = attr.ib()

    @property
    def geojson(self):
        return dict(
            type="FeatureCollection",
            features=[
                dict(
                    type="Feature",
                    properties={},
                    id=ident,
                    geometry=geometry.mapping(g),
                )
                for ident, g in zip(self.table.ident, self.table.geometry)
            ],
        )

    def name_state(self, counties):
        return name_state(
            self.regions,
            np.array(self.table.population),
            self.cities,
            counties,
        )

    def polygon(self, idx):
        return self.table.iloc[idx].geometry


class GeographySource(ABC):
    def unique_id(self):
        typ = type(self)
        return typ.__module__ + "." + typ.__name__

    @abstractmethod
    def version(self):
        pass

    @abstractmethod
    def geo_dataframe(self):
        pass

    @abstractmethod
    def additional_edges(self):
        pass

    @abstractmethod
    def regions(self, geo_df):
        pass

    @abstractmethod
    def max_states(self):
        pass

    @abstractmethod
    def atlas_types(self):
        pass

    def construct(self):
        return construct_geography(self)


@permacache(
    "statesbot/geographies/geography",
    key_function=dict(source=lambda x: [x.unique_id(), x.version()]),
)
def construct_geography(source):
    table = source.geo_dataframe()
    table["spherical_geometry"] = table.geometry.apply(
        lambda x: transform(to_spherical, x)
    )
    ident_back = {x: i for i, x in enumerate(table.ident)}
    cities = {ident_back[k]: v for k, v in classify_cities(table).items()}

    centers = np.array([list(polygon.centroid.coords)[0] for polygon in table.geometry])

    weights = dict(get_edges(table).items())
    weights.update({(v, u): w for (u, v), w in weights.items()})

    neighbors = [set() for _ in range(len(table))]
    for x, y in weights:
        neighbors[x].add(y)
    for x, y in source.additional_edges():
        x, y = ident_back[x], ident_back[y]
        assert x not in neighbors[y], str((x, y))
        neighbors[x].add(y)
        neighbors[y].add(x)
        weights[x, y] = weights[y, x] = 0
    return Geography(
        table=table,
        cities=cities,
        centers=centers,
        weights=weights,
        neighbors=neighbors,
        regions=source.regions(table),
        max_states=source.max_states(),
        atlas_types=source.atlas_types(),
    )


def to_spherical(lon, lat):
    lon, lat = lon / 180 * np.pi, lat / 180 * np.pi
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z]).T
