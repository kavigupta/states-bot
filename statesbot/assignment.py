from collections import defaultdict

import attr
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from shapely.ops import unary_union
from shapely import geometry

from .map import MapObject


@attr.s
class Assignment:
    data = attr.ib()
    meta = attr.ib()
    state_to_counties = attr.ib()
    county_to_state = attr.ib()

    @classmethod
    def from_county_to_state(cls, data, meta, county_to_state):
        state_to_counties = defaultdict(set)
        for county, state in enumerate(county_to_state):
            state_to_counties[state].add(county)
        return cls(data, meta, state_to_counties, county_to_state)

    @property
    def state_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(self.meta.count))

        for state in range(self.meta.count):
            for other_state in range(self.meta.count):
                if self.meta.bordering(
                    self.state_to_counties[other_state], self.state_to_counties[state]
                ):
                    graph.add_edge(state, other_state)
        return graph

    @property
    def coloring(self):
        rng = np.random.RandomState(0)
        g = self.state_graph
        coloring = rng.choice(6, size=len(self.state_to_counties))
        while True:
            resolved = True
            for v in range(coloring.shape[0]):
                while any(coloring[v] == coloring[n] for n in g.neighbors(v) if n != v):
                    u = rng.choice(coloring.shape[0])
                    coloring[u], coloring[v] = coloring[v], coloring[u]
                    resolved = False
            if resolved:
                return coloring.tolist()

    def draw(self, data, four_color=False):
        coloring = self.coloring
        cts = self.county_to_state
        if four_color:
            cts = [coloring[x if x == x else 0] + x / 100 for x in cts]
        figure = go.Figure(
            go.Choropleth(
                geojson=data.geojson,
                locations=data.idents,
                z=cts,
                marker_line_width=0,
                name="margin",
                showscale=False,
            )
        )
        figure.update_geos(scope="usa")
        # figure.update_layout(geo=dict(bgcolor=BACKGROUND, lakecolor=BACKGROUND))
        figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return figure

    def county_polygons_for_state(self, state):
        counties = self.state_to_counties[state]
        return [self.data.polygon(county) for county in counties]

    def multi_polygons_for_state(self, state):
        polys = unary_union([c for c in self.county_polygons_for_state(state)])
        if isinstance(polys, geometry.polygon.Polygon):
            return [polys]
        return polys

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
                    list(zip(*poly.exterior.coords.xy))
                    for poly in self.multi_polygons_for_state(state)
                ],
            },
        }

    def state_name(self, state, data):
        return data.name_state(self.state_to_counties[state])

    def export(self, data):
        return MapObject(
            coloring=self.coloring,
            states=list(self.state_to_counties),
            ident_to_state={
                x: int(self.county_to_state[i]) for i, x in enumerate(data.idents)
            },
            capitols={
                state: max(
                    [
                        city
                        for county in self.state_to_counties[state]
                        for _, city in data.cities[county].iterrows()
                    ],
                    key=lambda x: x.Population,
                )
                for state in self.state_to_counties
            },
            state_names={
                state: self.state_name(state, data) for state in self.state_to_counties
            },
            polygons={
                state: self.multi_polygons_for_state(state)
                for state in self.state_to_counties
            },
        )
