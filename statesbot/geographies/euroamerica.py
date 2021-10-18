from collections import defaultdict

import geopandas
import pandas as pd
import pycountry

from ..utils import sequence, to_all

from .geography import GeographySource
from .europe import EuropeCountiesDataset
from .usa import USACountiesDataset


class EuroAmericaCountiesDataset(GeographySource):
    euro = EuropeCountiesDataset()
    usa = USACountiesDataset()

    def version(self):
        return f"1.0.0 [euro={self.euro.version()}, usa={self.usa.version()}"

    def geo_dataframe(self):
        euro = self.euro.geo_dataframe()[
            ["ident", "NAME_LATN", "geometry", "population", "dem_2020"]
        ].rename(columns={"NAME_LATN": "name"})
        usa = self.usa.geo_dataframe()
        return pd.concat([euro, usa])

    def additional_edges(self):
        return [
            *self.euro.additional_edges(),
            *self.usa.additional_edges(),
            ("23003", "IS002"),
        ]

    def regions(self, geodb):
        return {
            **self.euro.regions(self.euro.geo_dataframe()),
            **self.usa.regions(self.usa.geo_dataframe()),
        }

    def atlas_types(self):
        return ["atlas_euroamerica"]

    def max_states(self):
        return 75
