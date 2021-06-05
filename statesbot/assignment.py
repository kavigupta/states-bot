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

    def assign(self, county, state):
        current_state = self.county_to_state[county]
        self.county_to_state[county] = state
        self.state_to_counties[current_state].remove(county)
        self.state_to_counties[state].add(county)

    @property
    def state_lists(self):
        return [list(x) for _, x in sorted(self.state_to_counties.items())]

    # @property
    # def centroids(self):
    #     return np.array([(centers[x] * pops[x][:,None]).sum(0) / pops[x].sum(0) for x in self.state_lists])
    #
    @property
    def aggregated_stats(self):
        return np.array([self.meta.stat[x].sum() for x in self.state_lists])

    @classmethod
    def from_county_to_state(cls, data, meta, county_to_state):
        state_to_counties = defaultdict(set)
        for county, state in enumerate(county_to_state):
            state_to_counties[state].add(county)
        return cls(data, meta, state_to_counties, county_to_state)

    def optimal_border_transitions(self):
        bordering_states = []
        for idxa, a in self.state_to_counties.items():
            for idxb, b in self.state_to_counties.items():
                if idxa == idxb:
                    continue
                if not self.meta.bordering(a, b):
                    continue
                bordering_states.append((idxa, idxb))
        bigger, smaller = np.array(bordering_states).T
        idx = np.argsort(
            self.aggregated_stats[bigger] - self.aggregated_stats[smaller]
        )[::-1]
        return np.array(bordering_states)[idx]

    def graph_for_state(self, bigger):
        big_graph = nx.Graph()
        big_graph.add_nodes_from(self.state_to_counties[bigger])
        for c1 in self.state_to_counties[bigger]:
            for c2 in self.meta.graph.neighbors[c1]:
                if c2 in self.state_to_counties[bigger]:
                    big_graph.add_edge(c1, c2)
        return big_graph

    def attempt_fix(self, bigger, smaller, current_iter):
        border_counties = {
            x
            for x in self.state_to_counties[bigger]
            if self.meta.graph.neighbors[x] & self.state_to_counties[smaller]
        }
        big_graph = self.graph_for_state(bigger)

        def transferability(county):
            neighbors = self.meta.graph.neighbors[county]
            adj_small = sum(
                self.meta.graph.weights[a, county]
                for a in neighbors & self.state_to_counties[smaller]
                if a != county
            )
            adj_big = sum(
                self.meta.graph.weights[a, county]
                for a in neighbors & self.state_to_counties[bigger]
                if a != county
            )
            return adj_small / (adj_big + 1e-10)

        bigger_stat, smaller_stat = self.aggregated_stats[[bigger, smaller]]
        progress = False

        counties = sorted(border_counties, key=transferability, reverse=True)
        for county in counties:
            tr = transferability(county)
            if current_iter < 300 and tr < 0.5:
                continue
            if tr < 0.25:
                continue
            if tr < 0.75 and bigger_stat / smaller_stat < 1.25:
                continue
            if county in nx.algorithms.components.articulation_points(big_graph):
                continue
            if self.meta.stat[county] * 2 < bigger_stat - smaller_stat:
                self.assign(county, smaller)
                bigger_stat -= self.meta.stat[county]
                smaller_stat += self.meta.stat[county]
                big_graph.remove_node(county)
                progress = True
        return progress

    def fix_border(self, itr):
        for bigger, smaller in self.optimal_border_transitions():
            if self.attempt_fix(bigger, smaller, itr):
                return True
        return False

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

    def remove_farthest(self, state_idx, rng):
        state = sorted(
            self.state_to_counties[state_idx]
            - set(
                nx.algorithms.components.articulation_points(
                    self.graph_for_state(state_idx)
                )
            )
        )

        coupling_by_county = {
            x: sum(
                self.meta.graph.weights.get((x, y), 0)
                for y in self.meta.graph.neighbors[x]
                if y in self.state_to_counties[state_idx]
            )
            for x in state
        }

        idx_farthest = min(state, key=coupling_by_county.get)

        contribution = self.meta.stat[idx_farthest] / self.meta.stat[state].sum()

        if contribution > 0.3:
            return

        if coupling_by_county[idx_farthest] > 0.3:
            return

        alternate_states = set(
            self.county_to_state[list(self.meta.graph.neighbors[idx_farthest])]
        ) - {state_idx}
        if not alternate_states:
            return
        self.assign(idx_farthest, rng.choice(list(alternate_states)))
        return True

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
                locations=[c.ident for c in data.countylikes],
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
        return [
            poly
            for county in counties
            for polys in self.data.countylikes[county].coordinates
            for poly in polys
        ]

    def multi_polygons_for_state(self, state):
        polys = unary_union(
            [geometry.Polygon(c) for c in self.county_polygons_for_state(state)]
        )
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
                x.ident: int(self.county_to_state[i])
                for i, x in enumerate(data.countylikes)
            },
            capitols={
                state: max(
                    [
                        city
                        for county in self.state_to_counties[state]
                        for city in data.countylikes[county].cities
                    ],
                    key=lambda x: x["population"],
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
