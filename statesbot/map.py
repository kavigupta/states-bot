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
    dem_2020 = attr.ib(default=None)
    pop_2020 = attr.ib(default=None)

    def attach_to(self, data):
        dem_2020_by_ident = {
            c.ident: (c.population, c.dem_2020 * c.population)
            for _, c in data.table.iterrows()
        }
        self.dem_2020 = {}
        self.pop_2020 = {}
        for state in self.states:
            pop = 0
            margin = 0
            for c, s in self.ident_to_state.items():
                if s != state:
                    continue
                p, m = dem_2020_by_ident[c]
                pop += p
                margin += m
            self.dem_2020[state] = margin / pop
            self.pop_2020[state] = pop

    def ship(self):
        coloring = self.coloring
        shape = fiona.open(
            "temporary/counties.shp",
            "w",
            "ESRI Shapefile",
            {
                "properties": {
                    "id": "int:10",
                    "statecolor": "int:10",
                    "name": "str",
                    "dem_2020": "float",
                },
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
            s = self.feature_for_state(state, c, self.state_names[state])
            shape.write(s)
            city = self.capitols[state]
            capitols.write(
                {
                    "type": "Feature",
                    "id": str(state),
                    "properties": {},
                    "geometry": {
                        "type": "Point",
                        "coordinates": (city.x, city.y),
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
                dem_2020=self.dem_2020[state],
            ),
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    list(zip(*poly.exterior.coords.xy)) for poly in self.polygons[state]
                ],
            },
        }

    def apportion(self, count=435):
        district_size = sum(self.pop_2020.values()) / 435
        apportion = {
            s: max(1, int(self.pop_2020[s] / district_size)) for s in self.pop_2020
        }
        while sum(apportion.values()) < count:
            s = max(
                self.pop_2020,
                key=lambda s: self.pop_2020[s] - district_size * apportion[s],
            )
            apportion[s] += 1
        return apportion

    def statistics(self):
        apportionment = self.apportion()
        dem_ec, gop_ec, dem_senate, gop_senate = 0, 0, 0, 0
        for s in self.pop_2020:
            if self.dem_2020[s] > 0:
                dem_ec += apportionment[s] + 2
                dem_senate += 2
            else:
                gop_ec += apportionment[s] + 2
                gop_senate += 2
        return dem_ec, gop_ec, dem_senate, gop_senate
