from collections import defaultdict
from functools import lru_cache
import json

import numpy as np
import geopandas
import shapely
from shapely.ops import unary_union

from permacache import permacache, stable_hash

from .counties import get_counties, Countylike

replacements = {
    "06037": "subcounty-geojsons/los-angeles/counties.shp",
    "06065": "subcounty-geojsons/riverside/CA_Counties_TIGER2016.shp",
    "06071": "subcounty-geojsons/san-bernardino/CA_Counties_TIGER2016.shp",
}


@lru_cache(None)
def precincts():
    with open("precinctdata/precinctdata.json") as f:
        return json.load(f)


def get_countylikes():
    result = []
    for c in get_counties():
        if c.ident not in replacements:
            result.append(c)
            continue
        result += get_subset_county(
            c, geopandas.read_file(replacements[c.ident]).to_crs(epsg=4326)
        )
    return result


def partition(feat, shapefile_df):
    subcounty_polys = defaultdict(list)
    for _, row in shapefile_df.iterrows():
        subcounty = row["SUBCOUNTY"]
        if subcounty is None:
            continue
        subcounty_polys[subcounty] += (
            [row.geometry]
            if isinstance(row.geometry, shapely.geometry.polygon.Polygon)
            else row.geometry
        )

    subcounty_polys = {
        k: close_holes(unary_union(subcounty_polys[k])).simplify(0.01)
        for k in subcounty_polys
    }
    feat_poly = shapely.geometry.MultiPolygon(
        [shapely.geometry.Polygon(x[0]) for x in feat.coordinates]
    )

    def snap(x, y, tol=0.02):
        return shapely.ops.unary_union(shapely.ops.snap(x, y, tol))

    snapped = {}
    for k in sorted(subcounty_polys, key=lambda x: -subcounty_polys[x].area):
        updated = snap(subcounty_polys[k], feat_poly, 0.04)
        for k2 in snapped:
            updated = snap(updated, snapped[k2], 0.04)
        snapped[k] = updated
    return snapped


@permacache(
    "statesbot/countylikes/get_subset_county",
    key_function=dict(
        feat=stable_hash,
        shape_df=lambda x: stable_hash(np.array(x.applymap(str)).tolist()),
    ),
)
def get_subset_county(feat, shape_df):
    subcounty_polys = partition(feat, shape_df)
    cities = classify_elements(
        subcounty_polys,
        feat.cities,
        lambda x: shapely.geometry.Point(x["longitude"], x["latitude"]),
    )
    precincts_by = classify_elements(
        subcounty_polys,
        [p for p in precincts() if p["GEOID"].startswith(feat.ident)],
        lambda x: shapely.geometry.Point(*x["centroid"]),
    )
    aggd_votes = {
        k: np.array(
            [
                [x["votes_dem"], x["votes_rep"], x["votes_total"]]
                for x in precincts_by[k]
            ]
        ).sum(0)
        for k in subcounty_polys
    }
    subcounties = []
    for subcounty in subcounty_polys:
        subcounties.append(
            Countylike(
                ident=feat.ident + "." + subcounty,
                cities=cities[subcounty],
                area=subcounty_polys[subcounty].area
                / sum(v.area for v in subcounty_polys.values())
                * feat.area,
                pop=aggd_votes[subcounty][-1]
                / sum(v[-1] for v in aggd_votes.values())
                * feat.pop,
                coordinates=coordinates_for(subcounty_polys[subcounty]),
                dem_2020=(aggd_votes[subcounty][0] - aggd_votes[subcounty][1])
                / aggd_votes[subcounty][2],
            )
        )
    return subcounties


def classify_elements(polys, elements, coord_fn):
    elements_each = defaultdict(list)
    for element in elements:
        for poly in polys:
            if polys[poly].contains(coord_fn(element)):
                elements_each[poly].append(element)
                break
        else:
            print("Could not classify", element)
    return elements_each


def close_holes(poly):
    if isinstance(poly, shapely.geometry.Polygon):
        return shapely.geometry.Polygon(poly.exterior)
    return shapely.geometry.MultiPolygon([close_holes(p) for p in poly])


def coordinates_for(poly):
    if isinstance(poly, shapely.geometry.Polygon):
        poly = [poly]
    return [[list(zip(*poly.exterior.xy))] for poly in poly]
