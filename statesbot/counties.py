from collections import defaultdict

import pandas as pd
import numpy as np

import attr
import us
import geonamescache

import electiondata as e
from electiondata.examples.census_2010_pop import Census2010Population
from electiondata.examples.plotly_geojson import PlotlyGeoJSON


@attr.s
class Countylike:
    ident = attr.ib()
    cities = attr.ib()
    area = attr.ib()
    coordinates = attr.ib()
    pop = attr.ib()

    @classmethod
    def of(cls, f, city_map, pop_by_fips):
        return cls(
            ident=f["id"],
            cities=city_map[f["id"]],
            area=f["properties"]["CENSUSAREA"],
            coordinates=f["geometry"]["coordinates"],
            pop=pop_by_fips[f["id"]],
        )

    @property
    def state(self):
        return us.states.lookup(self.ident[:2]).name

    @property
    def feature(self):
        return dict(
            type="Feature",
            properties={},
            id=self.ident,
            geometry=dict(type="MultiPolygon", coordinates=self.coordinates),
        )

    @property
    def flat_coords(self):
        return np.array([x for x in self.coordinates for x in x for x in x])

    @property
    def center(self):
        return self.flat_coords.mean(0)


def get_counties():
    geojson = PlotlyGeoJSON().get()
    city_map = cities_dataset()
    data = Census2010Population(e.alaska.FIPS).get()
    pop_by_fips = dict(zip(data.FIPS, data.CENSUS2010POP))
    return [
        Countylike.of(f, city_map, pop_by_fips)
        for f in geojson["features"]
        if f["id"][:2] not in {"72", "15", "02"} and f["id"] not in {"25019", "53055"}
    ]


def get_countylikes():
    return get_counties()


def cities_dataset():
    cities_dataset = pd.read_csv(
        "https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv",
        sep="|",
    )
    cities_dataset = cities_dataset[cities_dataset.County == cities_dataset.County]
    normalizer = e.usa_county_to_fips("State short")
    normalizer.rewrite["hoonah angoon"] = "hoonah-angoon"
    normalizer.rewrite["matanuska susitna"] = "matanuska-susitna"
    normalizer.rewrite["prince of wales hyder"] = "prince of wales-hyder"
    normalizer.rewrite["valdez cordova"] = "valdez-cordova"
    normalizer.rewrite["yukon koyukuk"] = "yukon-koyukuk"
    normalizer.rewrite["saint louis"] = "saint louis city"
    normalizer.rewrite["northern mariana islands"] = "northern islands"
    normalizer.rewrite["baltimore"] = "baltimore city"
    normalizer.rewrite["saint thomas"] = "saint thomas island"
    normalizer.rewrite["franklin"] = "franklin city"
    normalizer.rewrite["richmond"] = "richmond city"
    normalizer.rewrite["saint croix"] = "saint croix island"
    normalizer.rewrite["bedford"] = "bedford city"
    normalizer.rewrite["fairfax"] = "fairfax city"
    normalizer.rewrite["roanoke"] = "roanoke city"
    normalizer.rewrite["saint john"] = "saint john island"
    normalizer.rewrite["american samoa"] = "ERROR"
    normalizer.rewrite["federated states of micro"] = "ERROR"
    normalizer.rewrite["marshall islands"] = "ERROR"
    normalizer.rewrite["palau"] = "ERROR"

    normalizer.apply_to_df(cities_dataset, "County", "FIPS", var_name="normalizer")
    cd = e.remove_errors(cities_dataset, "FIPS")

    result = {}
    for city, state, fips in zip(cd.City, cd["State short"], cd.FIPS):
        if (city, state) in result:
            continue
        result[city, state] = fips

    usa = [
        city
        for city in geonamescache.GeonamesCache().get_cities().values()
        if city["countrycode"] == "US"
    ]
    cities = defaultdict(list)
    for city in usa:
        key = city["name"].replace("St.", "Saint"), city["admin1code"]
        if key not in result:
            continue
        cities[result[key]].append(city)
    return cities
