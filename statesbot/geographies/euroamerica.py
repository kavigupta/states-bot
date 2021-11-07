from collections import defaultdict

import geopandas
import pandas as pd
import pycountry

from statesbot.geographies.greenland import GreenlandDataset

from ..utils import sequence, to_all

from .geography import CombinedGeographySource
from .europe import EuropeCountiesDataset
from .north_america import NorthAmericaDataset


class EuroAmericaCountiesDataset(CombinedGeographySource):
    def elements(self):
        return dict(
            euro=EuropeCountiesDataset(),
            usa=NorthAmericaDataset(),
        )

    def combined_version(self):
        return "1.0.1"

    def glue_edges(self):
        return [("GREENLAND", "IS002")]

    def atlas_types(self):
        return ["atlas_euroamerica"]

    def max_states(self):
        return 75
