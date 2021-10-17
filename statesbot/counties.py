from collections import defaultdict

import pandas as pd
import numpy as np

import attr
import us
import geonamescache

import electiondata as e
from electiondata.examples.census_2010_pop import Census2010Population
from electiondata.examples.plotly_geojson import PlotlyGeoJSON

from shapely import geometry


@attr.s
class Countylike:
    ident = attr.ib()
    cities = attr.ib()
    polygon = attr.ib()
    pop = attr.ib()
    dem_2020 = attr.ib()

    @property
    def state(self):
        return us.states.lookup(self.ident[:2]).name

    @property
    def feature(self):
        return dict(
            type="Feature",
            properties={},
            id=self.ident,
            geometry=geometry.mapping(self.polygon),
        )

    @property
    def center(self):
        return list(self.polygon.centroid.coords)[0]
