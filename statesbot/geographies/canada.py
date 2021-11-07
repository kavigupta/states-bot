from collections import defaultdict

import geopandas
import pandas as pd
import pycountry
from permacache import permacache

from .geography import GeographySource

from ..utils import to_all


@permacache("statesbot/canada/canada_dataset")
def canada_dataset():
    df = pd.read_csv("csvs/canada_census.csv", low_memory=False)
    df = df[df["DIM: Profile of Census Divisions (2247)"] == "Population, 2016"]
    populations = dict(
        zip(df["GEO_CODE (POR)"], df["Dim: Sex (3): Member ID: [1]: Total - Sex"])
    )
    canada = geopandas.read_file("shapefiles/canada_counties/lcd_000b16a_e.shp").to_crs(
        epsg=4326
    )
    canada.geometry = canada.geometry.simplify(tolerance=0.01)
    canada["population"] = canada["CDUID"].apply(lambda x: int(populations[int(x)]))
    canada["ident"] = canada["CDUID"]
    canada["dem_2020"] = 0
    return canada[["ident", "geometry", "population", "dem_2020", "PRNAME"]]


class CanadaDataset(GeographySource):
    def version(self):
        return "1.0.5"

    def geo_dataframe(self):
        return canada_dataset()

    def additional_edges(self):
        return [
            # Newfoundland to Nova Scotia
            ("1003", "1218"),
            # Nefoundland to Labrador
            *to_all("1009", "2498", "1010"),
            # Nova Scotia to Mainland
            *to_all("1101", "1212", "1214"),
            *to_all("1102", "1211", "1212"),
            *to_all("1103", "1307", "1308"),
            # Magdalen Islands to mainland
            ("1214", "2401"),
            ## Across Saint Lawrence
            #   Lile dOrelans
            *to_all("2420", "2419", "2421", "2423", "2425"),
            #   In between islands
            ("2423", "2425"),
            ("2433", "2434"),
            ("2437", "2438"),
            *to_all("2452", "2453", "2459"),
            ("2460", "2459"),
            #   Montreal to mainland
            *to_all("2466", "2467", "2458"),
            #   Montreal to Laval and Vaudreuil-Soulanges
            *to_all("2466", "2465", "2471"),
            #   Laval to mainland
            *to_all("2465", "2473"),
            #   West of Montreal
            # already has ("2470", "2471"),
            ("2469", "3501"),
            # Manitoulin to mainland
            ("3551", "3541"),
            # Prince Edward, not the province
            *to_all("3513", "3512", "3511"),
        ]

    def regions(self, geodb):
        by_state = defaultdict(list)
        for i, (_, row) in enumerate(geodb.iterrows()):
            by_state[row.PRNAME.split("/")[0]].append(i)
            by_state["Canada"].append(i)
        return dict(by_state.items())

    def atlas_types(self):
        return ["atlas_canada"]

    def max_states(self):
        return 5
