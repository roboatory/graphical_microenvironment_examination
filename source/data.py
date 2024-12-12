import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

def prepare_data(graphs, mapping, label):
    """convert a dictionary of microenvironments to a list of torch data objects
    
    args:
        graphs: a dictionary mapping sample names to subgraph data
        mapping: a dictionary mapping node labels to feature indices for one-hot encoding
        label: the graph-level label (0 for center, 1 for edge)
    """

    data_list = []

    for _, microenvironments in graphs.items():
        for microenvironment in microenvironments:
            microenvironment = nx.relabel_nodes(microenvironment, {node: n for n, node in enumerate(microenvironment.nodes())})
            
            node_features = []
            for node in microenvironment.nodes():
                features = np.zeros(len(mapping))
                features[mapping[microenvironment.nodes[node]["label"]]] = 1
                node_features.append(features)
            
            data = Data(
                x = torch.tensor(np.array(node_features), dtype = torch.float),
                y = torch.tensor([label], dtype = torch.long),
                edge_index = torch.tensor(list(microenvironment.edges), dtype = torch.long).t().contiguous()
            )

            data_list.append(data)

    return data_list