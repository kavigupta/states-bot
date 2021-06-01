import attr

import fiona


@attr.s
class MapObject:
    coloring = attr.ib()
    states = attr.ib()
    ident_to_state = attr.ib()
    capitols = attr.ib()
    state_names = attr.ib()
    polygons = attr.ib()

    def ship(self):
        coloring = self.coloring
        shape = fiona.open(
            "temporary/counties.shp",
            "w",
            "ESRI Shapefile",
            {
                "properties": {"id": "int:10", "statecolor": "int:10", "name": "str"},
                "geometry": "Polygon",
            },
        )
        capitols = fiona.open(
            "temporary/capitol.shp",
            "w",
            "ESRI Shapefile",
            {"properties": {}, "geometry": "Point"},
        )

        for state in self.states:

            c = coloring[state]
            shape.write(self.feature_for_state(state, c, self.state_names[state]))
            city = self.capitols[state]
            capitols.write(
                {
                    "type": "Feature",
                    "id": str(state),
                    "properties": {},
                    "geometry": {
                        "type": "Point",
                        "coordinates": (city["longitude"], city["latitude"]),
                    },
                }
            )
        shape.close()
        capitols.close()

    def feature_for_state(self, state, statecolor, name):
        return {
            "type": "Feature",
            "id": str(state),
            "properties": dict(
                id=int(state),
                name=name,
                statecolor=int(statecolor),
            ),
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    list(zip(*poly.exterior.coords.xy)) for poly in self.polygons[state]
                ],
            },
        }
