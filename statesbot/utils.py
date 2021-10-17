import numpy as np

from permacache import stable_hash
from shapely import geometry


def distances(a, b):
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    assert len(a.shape) == len(b.shape) == 3
    lat1 = a[:, :, 1]
    lon1 = a[:, :, 0]
    lat2 = b[:, :, 1]
    lon2 = b[:, :, 0]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c


def hash_dataframe(df):
    return (
        stable_hash({k: list(df[k]) for k in df if "geometry" not in k}),
        stable_hash(
            {k: [geometry.mapping(x) for x in df[k]] for k in df if "geometry" in k}
        ),
    )
