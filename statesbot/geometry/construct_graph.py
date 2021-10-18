import tqdm.auto as tqdm
from permacache import permacache

import geopandas
from shapely import geometry
from shapely.ops import snap

from ..utils import hash_dataframe


@permacache(
    "statesbot/geometry/construct_graph/get_edges",
    key_function=dict(df=hash_dataframe),
)
def get_edges(df):
    ident_back = {ident: i for i, ident in enumerate(df.ident)}
    possible_edges = [
        (ident_back[row.ident_left], ident_back[row.ident_right])
        for _, row in geopandas.sjoin(df, df, how="left", op="intersects").iterrows()
        if row.ident_left != row.ident_right
    ]
    polygons = list(df.spherical_geometry)
    edges = {}
    print("EDGES")
    for i, j in tqdm.tqdm(possible_edges):
        intersection = (
            snap(polygons[i], polygons[j], 1e-4)
            .buffer(0)
            .intersection(polygons[j].buffer(0))
        )
        if intersection.is_empty:
            continue
        if isinstance(
            intersection, (geometry.point.Point, geometry.multipoint.MultiPoint)
        ):
            continue
        if isinstance(intersection, geometry.collection.GeometryCollection):
            intersection = geometry.multilinestring.MultiLineString(
                [
                    x
                    for x in intersection
                    if not isinstance(
                        x, (geometry.point.Point, geometry.polygon.Polygon)
                    )
                ]
            )
        multiplier = 1
        if isinstance(
            intersection,
            (geometry.polygon.Polygon, geometry.multipolygon.MultiPolygon),
        ):
            multiplier *= 0.5
            intersection = intersection.boundary
        assert isinstance(
            intersection,
            (
                geometry.multilinestring.MultiLineString,
                geometry.linestring.LineString,
            ),
        )
        edges[i, j] = intersection.length * multiplier
    return edges
