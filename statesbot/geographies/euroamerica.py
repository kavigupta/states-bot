from collections import defaultdict

import geopandas
import pandas as pd
import pycountry

from ..utils import sequence, to_all

from .geography import CombinedGeographySource
from .europe import EuropeCountiesDataset
from .usa import USACountiesDataset


class EuroAmericaCountiesDataset(CombinedGeographySource):
    def elements(self):
        return dict(
            euro=EuropeCountiesDataset(),
            usa=USACountiesDataset(),
        )

    def combined_version(self):
        return "1.0.1"

    def glue_edges(self):
        return [("23003", "IS002")]

    def atlas_types(self):
        return ["atlas_euroamerica"]

    def max_states(self):
        return 75
