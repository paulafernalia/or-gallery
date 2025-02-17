import networkx as nx
from typing import List, Tuple
from or_algorithms import dijkstra as djks


def solve_minimum_spanning_tree(graph: nx.Graph) -> nx.Graph:
    # Get edges
    edge_list = create_edge_list(graph)

    # Initialise minimum spanning tree with the nodes in the graph
    tree = initialise_tree(graph)

    # For each edge, starting from the one with lowest weight
    for e in edge_list:
        # Check if adding this edge would result in a cycle
        if not are_nodes_connected(tree, e[1], e[2]):
            # Add them this edge to the tree
            tree.add_edge(e[1], e[2], weight=e[0])

    return tree


def are_nodes_connected(graph: nx.Graph, node1: str, node2: str) -> nx.Graph:
    # Find the shortest path between the nodes we aim to connect
    _, path = djks.solve_shortest_path(graph, node1, node2)

    return path is not None


def initialise_tree(graph: nx.Graph) -> nx.Graph:
    tree = nx.Graph()
    for node in graph.nodes:
        tree.add_node(node)

    return tree


def create_edge_list(graph: nx.Graph) -> List[Tuple[int, str, str]]:
    edges = [(e[2]["weight"], e[0], e[1]) for e in graph.edges(data=True)]

    # Sort in descending order
    return sorted(edges)
