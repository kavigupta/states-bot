from collections import defaultdict
import json

import geopandas
import numpy as np
import pandas as pd
from permacache import permacache
import shapely.geometry as geometry

import addfips
import us

from .geography import GeographySource

COUNTIES_TO_SPLIT = {
    "06037",
    "17031",
    "48201",
    "04013",
    "06073",
    "06059",
    "36047",
    "12086",
    "48113",
    "06065",
    "36081",
    "53033",
    "32003",
    "06071",
    "48439",
    "48029",
    "12011",
    "06085",
}


def usa_counties_dataset():
    df = geopandas.read_file("shapefiles/usa_counties/cb_2020_us_county_500k.shp")
    df["FIPS"] = df.STATEFP + df.COUNTYFP
    df = df[df.FIPS.apply(lambda x: x[:2] not in {"78", "60", "69", "66"})]
    df = df[["FIPS", "NAME", "geometry"]]
    election = pd.read_csv("csvs/2020_demographics_votes_fips.csv", dtype={"FIPS": str})
    election["FIPS"] = election.FIPS.apply(lambda x: ("0" + x)[-5:])
    census = pd.read_csv(
        "https://raw.githubusercontent.com/kavigupta/census-downloader/master/outputs/counties_census2020.csv"
    )
    census["FIPS"] = census.STUSAB.apply(
        addfips.AddFIPS().get_state_fips
    ) + census.COUNTY.apply(lambda x: f"{int(x):03d}")
    populations = dict(zip(census.FIPS, census.POP100))
    df["population"] = df.FIPS.map(lambda x: populations[x])
    # https://en.wikipedia.org/wiki/2020_United_States_presidential_election_in_Hawaii
    biden_2020 = dict(zip(election.FIPS, election["Biden 2020 Margin"]))

    def get_biden_2020(fips):
        if fips[:2] == "72":
            # https://twitter.com/SageOfTime1/status/1382485916095283203?s=20
            return 0.618 - 0.375
        if fips in {"02066", "02063"}:
            fips = "02261"
        if fips == "02158":
            fips = "02270"
        if fips == "15005":
            return 0.9583 - 0.0417
        return biden_2020[fips]

    df["dem_2020"] = df.FIPS.map(get_biden_2020)
    return df


def split_county(county_row, subcounties, precinctdata):
    relevant_sc = subcounties[subcounties.county_fips == county_row.FIPS]
    polygons = list(relevant_sc.geometry)
    precincts = [[] for _ in range(len(polygons))]
    for precinct in precinctdata:
        if not precinct["GEOID"].startswith(county_row.FIPS):
            continue
        containing_polygons = [
            i
            for i, x in enumerate(polygons)
            if x.contains(geometry.point.Point(precinct["centroid"]))
        ]
        assert len(containing_polygons) < 2
        if len(containing_polygons) == 0:
            print("Could not classify", precinct)
            continue
        precincts[containing_polygons[0]].append(precinct)
    summaries = pd.DataFrame(
        [
            pd.DataFrame(p)[["votes_dem", "votes_rep", "votes_total"]].sum()
            for p in precincts
        ]
    )
    values = dict(
        FIPS=list(relevant_sc.county_fips + relevant_sc.COUSUBFP),
        NAME=list(relevant_sc.NAME),
        geometry=list(relevant_sc.geometry),
        population=(
            summaries.votes_total / summaries.votes_total.sum() * county_row.population
        )
        .round()
        .astype(np.int),
        dem_2020=(summaries.votes_dem - summaries.votes_rep) / summaries.votes_total,
    )
    return pd.DataFrame(values)


@permacache("statesbot/geographies/usa/usa_subcounties_dataset")
def usa_subcounties_dataset(version=7):
    subcounties = geopandas.read_file(
        "shapefiles/usa_subcounties/cb_2020_us_cousub_500k.shp"
    )
    subcounties["county_fips"] = subcounties.STATEFP + subcounties.COUNTYFP
    with open("precinctdata/precinctdata.json") as f:
        precinctdata = json.load(f)
    rows = []
    for _, row in usa_counties_dataset().iterrows():
        if row.FIPS in COUNTIES_TO_SPLIT:
            print(row.FIPS)
            rows += [
                x for _, x in split_county(row, subcounties, precinctdata).iterrows()
            ]
        else:
            rows += [row]
    df = pd.DataFrame(rows).rename(columns={"FIPS": "ident", "NAME": "name"})
    df = geopandas.GeoDataFrame(df, geometry=df.geometry)
    return df


class USACountiesDataset(GeographySource):
    def version(self):
        return "1.0.0"

    def geo_dataframe(self):
        return usa_subcounties_dataset()

    def additional_edges(self):
        return [
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

    def regions(self, geodb):
        by_state = defaultdict(list)
        for i, (_, row) in enumerate(geodb.iterrows()):
            by_state[us.states.lookup(row.ident[:2]).name].append(i)
        return dict(by_state.items())
