from collections import defaultdict

import geopandas
import pandas as pd
import pycountry

from ..utils import sequence, to_all

from .geography import CombinedGeographySource
from .canada import CanadaDataset
from .usa import USACountiesDataset
from .greenland import GreenlandDataset


class NorthAmericaDataset(CombinedGeographySource):
    def elements(self):
        return dict(
            euro=CanadaDataset(),
            usa=USACountiesDataset(),
            greenland=GreenlandDataset(),
        )

    def combined_version(self):
        return "1.0.1"

    def glue_edges(self):
        return [
            # UP to Ontario
            ("26033", "3551"),
            # Saint Laurence border
            *to_all("36045", "3510"),
            ("36089", "3501"),
        ]

    def remove_edges(self):
        return [("53073", "02198")]

    def atlas_types(self):
        return ["atlas_north_america"]

    def max_states(self):
        return 55
