import numpy as np
import networkx as nx
from permacache.hash import stable_hash


class Graph:
    def __init__(self, data, eqstat_key):
        self.nx_graph = nx.Graph()
        self.nx_graph.add_nodes_from(np.arange(len(data.table)))
        self.nx_graph.add_edges_from(
            [(u, v) for u, vs in enumerate(data.neighbors) for v in vs]
        )
        self.eqstat = np.array(data.table[eqstat_key])
        self.euclidean = data.centers
        self.weights = data.weights

    @property
    def vertex_indices(self):
        return np.arange(self.nx_graph.number_of_nodes())

    def total_eqstat(self, indices):
        return self.eqstat[indices].sum()

    def connected_components(self, subset):
        return nx.algorithms.components.connected_components(
            self.nx_graph.subgraph(subset)
        )

    def subset_connected(self, subset):
        return nx.algorithms.components.is_connected(self.nx_graph.subgraph(subset))

    def bordering(self, subset1, subset2):
        return self.subset_connected([*subset1, *subset2])

    @property
    def edges(self):
        return self.nx_graph.edges

    def neighbors(self, node):
        return self.nx_graph.neighbors(node)

    def weight(self, u, v):
        return self.weights[u, v]

    @property
    def hash(self):
        return dict(
            edges=stable_hash(nx.node_link_data(self.nx_graph)),
            eqstat=stable_hash(self.eqstat),
            weight=stable_hash(sorted(self.weights.items())),
            euclidean=stable_hash(self.euclidean),
        )
