from collections import defaultdict
import geopandas
import shapely
from shapely.ops import unary_union

from .counties import get_counties, Countylike

replacements = {"06037": "subcounty-geojsons/LA_County_City_Boundaries.shp"}


def get_countylikes():
    result = []
    for c in get_counties():
        if c.ident not in replacements:
            result.append(c)
            continue
        result += get_subset_county(c, replacements[c.ident])
    return result


def partition(feat, shape_path):
    df = geopandas.read_file(shape_path)
    df = df[df.FEAT_TYPE == "Land"]
    subcounty_pops = defaultdict(float)
    subcounty_areas = defaultdict(float)
    subcounty_cities = defaultdict(list)
    subcounty_polys = defaultdict(list)
    seen_cities = set()
    for _, row in df.iterrows():
        subcounty = row["SUBCOUNTY"]
        if subcounty is None:
            continue
        subcounty_areas[subcounty] += row.ShapeSTAre
        subcounty_polys[subcounty] += (
            [row.geometry]
            if isinstance(row.geometry, shapely.geometry.polygon.Polygon)
            else row.geometry
        )
        cities = [x for x in feat.cities if x["name"] == row.CITY_NAME]
        if not cities:
            continue
        assert len(cities) == 1, row.CITY_NAME
        [city] = cities
        subcounty_cities[subcounty].append(city)
        if row.CITY_NAME in seen_cities and row.CITY_NAME != "Unincorporated":
            continue
        seen_cities.add(row.CITY_NAME)
        subcounty_pops[subcounty] += city["population"]
    area_ratio = feat.area / sum(subcounty_areas.values())
    pop_ratio = feat.pop / sum(subcounty_pops.values())
    for k in subcounty_pops:
        subcounty_areas[k] *= area_ratio
        subcounty_pops[k] *= pop_ratio
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
    for k in subcounty_polys:
        updated = snap(subcounty_polys[k], feat_poly, 0.05)
        for k2 in snapped:
            updated = snap(updated, snapped[k2], 0.05)
        snapped[k] = updated
    return subcounty_cities, subcounty_areas, subcounty_pops, snapped


def get_subset_county(feat, shape_path):
    subcounty_cities, subcounty_areas, subcounty_pops, subcounty_polys = partition(
        feat, shape_path
    )
    subcounties = []
    for subcounty in subcounty_pops:
        subcounties.append(
            Countylike(
                ident=feat.ident + "." + subcounty,
                cities=subcounty_cities[subcounty],
                area=subcounty_areas[subcounty],
                pop=subcounty_pops[subcounty],
                coordinates=coordinates_for(subcounty_polys[subcounty]),
            )
        )
    return subcounties


def close_holes(poly):
    if isinstance(poly, shapely.geometry.Polygon):
        return shapely.geometry.Polygon(poly.exterior)
    return shapely.geometry.MultiPolygon([close_holes(p) for p in poly])


def coordinates_for(poly):
    if isinstance(poly, shapely.geometry.Polygon):
        poly = [poly]
    return [[list(zip(*poly.exterior.xy))] for poly in poly]
