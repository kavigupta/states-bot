from collections import defaultdict
import itertools

import attr
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from shapely.ops import unary_union
from shapely import geometry

from .map import MapObject


@attr.s
class Assignment:
    geography = attr.ib()
    graph = attr.ib()
    num_states = attr.ib()
    state_to_counties = attr.ib()
    county_to_state = attr.ib()

    @classmethod
    def from_county_to_state(cls, geography, graph, num_states, county_to_state):
        state_to_counties = defaultdict(set)
        for county, state in enumerate(county_to_state):
            state_to_counties[state].add(county)
        return cls(geography, graph, num_states, state_to_counties, county_to_state)

    @property
    def state_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_states))

        for state in range(self.num_states):
            for other_state in range(self.num_states):
                if self.graph.bordering(
                    self.state_to_counties[other_state], self.state_to_counties[state]
                ):
                    graph.add_edge(state, other_state)
        return graph

    @property
    def coloring(self):
        g = self.state_graph
        colors = nx.algorithms.coloring.greedy_color(g)
        if len(set(colors.values())) <= 6:
            return [colors[i] for i in range(len(self.state_to_counties))]
        for seed in itertools.count():
            if seed > 0:
                print("Coloring", seed)
            rng = np.random.RandomState(seed)
            coloring = rng.choice(6, size=len(self.state_to_counties))
            for _ in range(10 ** 4):
                resolved = True
                for v in range(coloring.shape[0]):
                    while any(
                        coloring[v] == coloring[n] for n in g.neighbors(v) if n != v
                    ):
                        u = rng.choice(coloring.shape[0])
                        coloring[u], coloring[v] = coloring[v], coloring[u]
                        resolved = False
                if resolved:
                    return coloring.tolist()

    def draw(self, four_color=False):
        coloring = self.coloring
        cts = self.county_to_state
        if four_color:
            cts = [coloring[x if x == x else 0] + x / 100 for x in cts]
        figure = go.Figure(
            go.Choropleth(
                geojson=self.geography.geojson,
                locations=self.geography.table.ident,
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
        return [self.geography.polygon(county) for county in counties]

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

    def state_name(self, state):
        return self.geography.name_state(self.state_to_counties[state])

    def export(self):
        return MapObject(
            coloring=self.coloring,
            states=list(self.state_to_counties),
            ident_to_state={
                x: int(self.county_to_state[i])
                for i, x in enumerate(self.geography.table.ident)
            },
            capitols={
                state: max(
                    [
                        city
                        for county in self.state_to_counties[state]
                        for _, city in self.geography.cities[county].iterrows()
                    ],
                    key=lambda x: x.Population,
                )
                for state in self.state_to_counties
            },
            state_names={
                state: self.state_name(state) for state in self.state_to_counties
            },
            polygons={
                state: self.multi_polygons_for_state(state)
                for state in self.state_to_counties
            },
        )
