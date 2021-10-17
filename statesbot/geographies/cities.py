import ast

import pandas as pd
import geopandas

from shapely import geometry

from permacache import permacache, stable_hash

from ..utils import hash_dataframe

CITY_COLUMNS = [
    "Name",
    "ASCII Name",
    "Alternate Names",
    "Country Code",
    "Population",
    "Elevation",
    "Timezone",
]


def cities_dataset():
    cities = pd.read_csv("csvs/cities_list.csv", sep=";")[
        [
            *CITY_COLUMNS,
            "Coordinates",
        ]
    ]
    cities = geopandas.GeoDataFrame(
        cities,
        geometry=cities.Coordinates.apply(
            lambda x: geometry.point.Point(*reversed(ast.literal_eval(x)))
        ),
    )
    cities["x"] = cities.geometry.map(lambda x: x.x)
    cities["y"] = cities.geometry.map(lambda x: x.y)
    return cities


@permacache(
    "statesbot/geographies/cities/classify_cities",
    key_function=dict(df=hash_dataframe),
)
def classify_cities(df, version=3):
    cities = cities_dataset()
    result = geopandas.sjoin(cities, df, how="left", op="intersects")
    cities_lists = {ident: [] for ident in df.ident}
    for _, row in result[result.ident == result.ident].iterrows():
        cities_lists[row.ident].append(row[[*CITY_COLUMNS, "x", "y"]])
    cities_lists = {k: pd.DataFrame(v) for k, v in cities_lists.items()}
    return cities_lists
