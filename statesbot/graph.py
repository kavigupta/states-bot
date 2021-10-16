import numpy as np
import networkx as nx
from permacache.hash import stable_hash


class Graph:
    """
    TODO replace the guts of this with something reasonable. right now just a wrapper
    """

    def __init__(self, data):
        self.data = data
        self.nx_graph = nx.Graph()
        self.nx_graph.add_nodes_from(self.vertex_indices)
        self.nx_graph.add_edges_from(
            [(u, v) for u, vs in enumerate(self.data.neighbors) for v in vs]
        )

    @property
    def vertex_indices(self):
        return np.arange(self.data.pops.size)

    @property
    def eqstat(self):
        return self.data.pops

    def total_eqstat(self, indices):
        return self.eqstat[indices].sum()

    @property
    def euclidean(self):
        return self.data.centers

    def connected_components(self, subset):
        return nx.algorithms.components.connected_components(
            self.nx_graph.subgraph(subset)
        )

    def subset_connected(self, subset):
        return nx.algorithms.components.is_connected(self.nx_graph.subgraph(subset))

    @property
    def edges(self):
        return self.nx_graph.edges

    def neighbors(self, node):
        return self.nx_graph.neighbors(node)

    def weight(self, u, v):
        return self.data.weights[u, v]

    @property
    def hash(self):
        return dict(
            edges=stable_hash([sorted(v) for v in self.data.neighbors]),
            eqstat=stable_hash(self.eqstat),
            weight=stable_hash(sorted(self.data.weights.items())),
        )
