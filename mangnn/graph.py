import json
import numpy as np
import torch


DATA_PATH = "mangnn/data/twitter/networks/networks.json"


class GraphLoader:
    """Graph loader."""

    def __init__(self, data_path=None):
        """Constructor for `GraphLoader`.

        Args:
            data_path: Path of graphs.
        """
        self.data_path = data_path if data_path else DATA_PATH
        self._graphs = None


    @property
    def graphs(self):
        """Return list of graphs."""
        if self._graphs is None:
            self._build_graphs()
        return self._graphs


    def _build_graphs(self):
        """Read graphs from file.

        Returns:
            graphs: List of graphs.
        """
        f = open(self.data_path)
        graphs = json.load(f)
        f.close()

        self._graphs = []
        for graph in graphs:
            graph["adj"] = self._load_sparse(graph["adj"])
            self._graphs.append(Graph(**graph))

        return self._graphs


    def _load_sparse(self, data_path):
        """Load sparse tensor.

        Args:
            data_path: Path of sparse data.

        Returns:
            Sparse tensor.
        """
        indices, values, size = np.load(data_path, allow_pickle=True)
        return torch.sparse_coo_tensor(indices=list(zip(*indices)), values=values, size=list(size))


class Graph:
    """ Graph. """
    
    def __init__(self, adj, src_type, tgt_type, agg_src, agg_tgt, name):
        """ Constructor for `Graph`.

        Args:
            adj: Adjacency matrix of graph connections.
            src_type: Type of source node ('user' or 'tag').
            tgt_type: Type of target node ('user' or 'tag').
            agg_src: Whether to aggregate messages from the source node to the target node.
            agg_tgt: Whether to aggregate messages from the target node to the source node.
            name: Graph name.
        """
        self.adj = adj
        self.src_type = src_type
        self.tgt_type = tgt_type
        self.agg_src = agg_src
        self.agg_tgt = agg_tgt
        self.name = name


    def get_relation_size(self):
        """ Return the total number of relations in the graph. """
        return self.adj._values().size()[0]


    def get_src_size(self):
        """ Return the total number of source nodes in the graph. """
        return self.adj.size()[0]


    def get_tgt_size(self):
        """ Return the total number of target nodes in the graph. """
        return self.adj.size()[1]