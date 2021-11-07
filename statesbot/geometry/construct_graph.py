import tqdm.auto as tqdm
from permacache import permacache, stable_hash

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
        if row.ident_left < row.ident_right
    ]
    polygons = list(df.spherical_geometry)
    edges = {}
    print("EDGES")
    for i, j in tqdm.tqdm(possible_edges):
        import numpy as np

        print(np.array(df.ident)[i], np.array(df.ident)[j])
        el = edge_length(polygons[i], polygons[j])
        if el is None:
            continue
        edges[i, j] = el
    return edges


@permacache(
    "statesbot/geometry/consturct_graph/edge_length",
    key_function=dict(
        a=lambda x: stable_hash(geometry.mapping(x)),
        b=lambda x: stable_hash(geometry.mapping(x)),
    ),
)
def edge_length(a, b):
    print("start")
    intersection = snap(a, b, 1e-4).buffer(0).intersection(a.buffer(0))
    print("done with intersection")
    if intersection.is_empty:
        return
    if isinstance(intersection, (geometry.point.Point, geometry.multipoint.MultiPoint)):
        return
    if isinstance(intersection, geometry.collection.GeometryCollection):
        intersection = geometry.multilinestring.MultiLineString(
            [
                x
                for x in intersection
                if not isinstance(x, (geometry.point.Point, geometry.polygon.Polygon))
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
    return intersection.length * multiplier
