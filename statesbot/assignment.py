from collections import defaultdict

import attr
import numpy as np
import networkx as nx
import plotly.graph_objects as go


@attr.s
class Assignment:
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
    def from_county_to_state(cls, meta, county_to_state):
        state_to_counties = defaultdict(set)
        for county, state in enumerate(county_to_state):
            state_to_counties[state].add(county)
        return cls(meta, state_to_counties, county_to_state)

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

    def attempt_fix(self, bigger, smaller):
        border_counties = {
            x
            for x in self.state_to_counties[bigger]
            if self.meta.graph.neighbors[x] & self.state_to_counties[smaller]
        }
        big_graph = self.graph_for_state(bigger)

        def transferability(county):
            neighbors = self.meta.graph.neighbors[county]
            return len(neighbors & self.state_to_counties[smaller]) / len(
                neighbors & self.state_to_counties[bigger]
            )

        bigger_stat, smaller_stat = self.aggregated_stats[[bigger, smaller]]
        progress = False
        for county in sorted(border_counties, key=transferability, reverse=True):
            if transferability(county) < 0.5:
                break
            if county in nx.algorithms.components.articulation_points(big_graph):
                continue
            if self.meta.stat[county] * 2 < bigger_stat - smaller_stat:
                self.assign(county, smaller)
                bigger_stat -= self.meta.stat[county]
                smaller_stat += self.meta.stat[county]
                big_graph.remove_node(county)
                progress = True
        return progress

    def fix_border(self):
        for bigger, smaller in self.optimal_border_transitions():
            if self.attempt_fix(bigger, smaller):
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

    def remove_farthest(self, state_idx):
        eles = self.meta.centers[sorted(self.state_to_counties[state_idx])]
        centroid = eles.mean(0)
        mean_dist = (((eles - centroid) ** 2).sum(1) ** 0.5).mean(0)
        state = sorted(
            self.state_to_counties[state_idx]
            - set(
                nx.algorithms.components.articulation_points(
                    self.graph_for_state(state_idx)
                )
            )
        )
        idx_farthest = state[
            ((self.meta.centers[state] - centroid) ** 2).sum(-1).argmax()
        ]

        contribution = self.meta.stat[idx_farthest] / self.meta.stat[state].sum()

        if contribution > 0.3:
            return

        real_dist = ((self.meta.centers[idx_farthest] - centroid) ** 2).sum() ** 0.5
        if real_dist / mean_dist <= 2.5:
            return

        alternate_states = set(
            self.county_to_state[list(self.meta.graph.neighbors[idx_farthest])]
        ) - {state_idx}
        if not alternate_states:
            return
        best = min(alternate_states, key=lambda i: self.aggregated_stats[i])
        self.assign(idx_farthest, best)
        return True

    def draw(self, data, four_color=False):
        coloring = nx.algorithms.coloring.greedy_color(self.state_graph)
        cts = self.county_to_state
        if four_color:
            cts = [coloring[x if x == x else 0] + x / 100 for x in cts]
        figure = go.Figure(
            go.Choropleth(
                geojson=data.geojson,
                locations=data.fipses,
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
