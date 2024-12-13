from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import voronoi_plot_2d

def plot_voronoi_diagram(voronoi, assignments, ax):
    """constructs a voronoi diagram with regions colored according to the provided cell type assignments
    
    args:
        voronoi (voronoi): a voronoi object containing the voronoi diagram data
        assignments (dataframe): a dataframe containing cell type assignments under 'FINAL_CELL_TYPE'
        ax (axes): a matplotlib axes object where the voronoi diagram will be plotted
    """

    unique_cell_types = np.unique(assignments["FINAL_CELL_TYPE"])
    
    colormap = plt.get_cmap("tab20c")
    colormap = {label: colormap(l) for l, label in enumerate(unique_cell_types)}
    
    for i, region in enumerate(voronoi.point_region):
        if -1 not in voronoi.regions[region]:
            polygon = [voronoi.vertices[r] for r in voronoi.regions[region]]
            ax.fill(*zip(*polygon), color = colormap[assignments["FINAL_CELL_TYPE"][i]], alpha = 0.5)
            
    voronoi_plot_2d(voronoi, ax, show_points = False, show_vertices = False, line_alpha = 0.5)

    ax.set_aspect("equal")
    ax.set_xlim(assignments["X"].min(), assignments["X"].max())
    ax.set_ylim(assignments["Y"].min(), assignments["Y"].max())

def plot_delaunay_triangulation(delaunay, assignments, ax):
    """plots a delaunay triangulation with points colored according to the provided cell type assignments

    args:
        delaunay (delaunay): a delaunay object containing the triangulation data
        assignments (dataframe): a dataframe containing cell type assignments under 'FINAL_CELL_TYPE'
        ax (axes): a matplotlib axes object where the delaunay triangulation will be plotted
    """

    unique_cell_types = np.unique(assignments["FINAL_CELL_TYPE"])
    
    colormap = plt.get_cmap("tab20c")
    colormap = {label: colormap(l) for l, label in enumerate(unique_cell_types)}
    
    ax.triplot(assignments["X"], assignments["Y"], delaunay.simplices, color = "black")
    ax.scatter(assignments["X"], assignments["Y"], c = [colormap[label] for 
                   label in assignments["FINAL_CELL_TYPE"]], edgecolor = "black", s = 100)
    
    legend_elements = [Patch(facecolor = colormap[label], edgecolor = "black", label = label) for label in unique_cell_types]
    ax.legend(handles = legend_elements, title = "cell types", loc = "center left", bbox_to_anchor = (1.02, 0.5))

    ax.set_aspect("equal")
    ax.set_xlim(assignments["X"].min(), assignments["X"].max())
    ax.set_ylim(assignments["Y"].min(), assignments["Y"].max())

def plot_microenvironment(microenvironment, center_node, ax):
    """plots a microenvironment with nodes colored according to their cell type assignments

    args:
        microenvironment (graph): a networkx graph representing the microenvironment
        center_node (int): the node in the microenvironment to use as the center
        ax (axes): a matplotlib axes object where the microenvironment will be plotted
    """

    positions = nx.get_node_attributes(microenvironment, "pos")
    labels = nx.get_node_attributes(microenvironment, "label")

    unique_cell_types = np.unique(list(labels.values()))
    
    colormap = plt.get_cmap("tab20c")
    colormap = {label: colormap(l) for l, label in enumerate(unique_cell_types)}
    node_colors = [colormap[labels[node]] for node in microenvironment.nodes()]

    nx.draw(microenvironment, positions, node_color = node_colors, node_size = 300, edge_color = "gray", ax = ax)

    legend_elements = [Patch(facecolor = colormap[label], edgecolor = "black", label = label) for label in unique_cell_types]
    ax.legend(handles = legend_elements, title = "cell types", loc = "center left", bbox_to_anchor = (1.02, 0.5))

    nx.draw_networkx_nodes(microenvironment, positions, nodelist = [center_node], node_color = "none", edgecolors = "red", linewidths = 2, ax = ax)

def construct_sample_graph(delaunay, assignments):
    """constructs a sample graph from the delaunay triangulation and cell type assignments

    args:
        delaunay (delaunay): a delaunay object containing the triangulation data
        assignments (dataframe): a dataframe containing cell type assignments under 'FINAL_CELL_TYPE'
    """

    graph = nx.Graph()
    
    points = assignments[["X", "Y"]].values
    for i, cell_centroid in enumerate(points):
        graph.add_node(i, pos = tuple(cell_centroid), label = assignments["FINAL_CELL_TYPE"][i])
        
    for simplex in delaunay.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                graph.add_edge(simplex[i], simplex[j])

    return graph

def construct_microenvironments(graphs, microenvironment_size = 3):
    """
    constructs microenvironments from the provided graphs

    args:
        graphs (dict): a dictionary where keys are sample identifiers and values are dictionaries containing networkx graphs under the key 'graph'
        microenvironment_size (int, optional): the radius of the microenvironment to construct; \
                                               defaults to 3
    """

    three_hop_graphs = {sample: [] for sample in graphs}
    
    for sample, data in graphs.items():
        G = data["graph"]
        for node in G.nodes():
            three_hop_graphs[sample].append(nx.ego_graph(G, node, radius = microenvironment_size))
            
    return three_hop_graphs