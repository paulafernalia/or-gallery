import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy.spatial.distance import euclidean
from typing import Tuple, List


def create_graph(nnodes: int, mapsize: int, seed: int=42) -> nx.Graph:
    """
    Generates a dense, undirected graph of a given number of nodes nnodes,
    randomly distributed throughout a grid of size mapsize x mapsize
    with Euclidean distances weights.

    Args:
        nnodes (int): Number of nodes in the graph.
        mapsize (int): Size of the grid.
        seed (int, optional): The random seed (default is 42).

    Returns:
        nx.Graph: A NetworkX graph where nodes have attribute 'x'
        (coordinates) and edges have an attribute 'w' (Euclidean distance).

    Raises:
        ValueError: If the provided seed is negative.
    """
    if seed < 1:
        raise ValueError("Seed must be positive")

    # Create n random points in 2D
    random.seed(seed)
    nodes = list(range(nnodes))

    # Define random locations
    coords = [
        (random.randint(0, mapsize), random.randint(0, mapsize))
        for i in nodes
    ]

    # Make the first point the depot, centred in the grid
    coords[0] = (int(mapsize / 2), int(mapsize / 2))

    # Calculate Euclidean distance between pairs of points (i != j)
    distances = {
        (i, j): euclidean(coords[i], coords[j])
        for i, j in combinations(nodes, 2)
    }

    # Create graph
    G: nx.Graph = nx.Graph()

    # Add nodes with coordinates and demand
    for i in nodes:
        G.add_node(i, x=coords[i])

    # Add edges with Manhattan distance as weights
    for (i, j), dist in distances.items():
        G.add_edge(i, j, d=dist)

    return G


def create_hamiltonian_tour(G: nx.Graph, seed: int=42) -> Tuple[nx.Graph, float]:
    """
    Create a Hamiltonian tour for the given graph.

    A Hamiltonian tour is a cycle that visits each vertex exactly once
    and returns to the starting vertex. This function attempts to find
    such a tour in the provided graph.

    Args:
        graph (dict): A dictionary representing the graph where keys are
            vertex identifiers and values are lists of adjacent vertices.
        seed (int, optional): The random seed (default value 42)

    Returns:
        nx.Graph: A networkx graph representing the Hamiltonian tour if
            one exists, otherwise an empty list.

        float: Total length of the generated cycle
    """
    random.seed(seed)
    
    # Choose a starting node
    start_node = random.choice(list(G.nodes))
    
    # Find the node that is furthest away
    far_edge = max(
        G.edges(start_node, data="d"), key=lambda e : e[2]
    )

    # Get the node that is furthest away
    far_node = far_edge[0] if far_edge[0] != start_node else far_edge[1]
    
    # Initialise cycle
    cycle = nx.Graph()
    cycle.add_node(start_node, x=G.nodes[start_node]["x"])
    cycle.add_node(far_node, x=G.nodes[far_node]["x"])
    cycle.add_weighted_edges_from([far_edge], weight="d")

    # Initialise total distance of the cycle
    total_dist = far_edge[2] * 2
    
    # Create a list of unvisited nodes
    unvisited = set(i for i in G.nodes if i not in cycle.nodes)
    
    # While there are nodes to insert
    while len(unvisited) > 0:
        # Find the cheapest insertion into current cycle
        best_n = None
        best_cost = math.inf
        best_e = None
    
        for n in unvisited:
            # For each edge in the cyle, try to insert n
            for i, j, d in cycle.edges(data='d'):
                cost = G.edges[i, n]["d"] + G.edges[n, j]["d"] - d
    
                assert cost > -1e-5
    
                # If it improves best insertion found so far
                if cost < best_cost:
                    best_n = n
                    best_cost = cost
                    best_e = (i, j)
    
        # Insert new edges
        new_edges = [(i, best_n, G.edges[i, best_n]['d']) for i in best_e]
        
        cycle.add_node(best_n, x=G.nodes[best_n]["x"])
        cycle.add_weighted_edges_from(new_edges, weight="d")

        # Update total distance
        total_dist += best_cost
    
        # Remove current edge
        if len(cycle.edges) > 3:
            cycle.remove_edge(*best_e)
    
        # Remove best node form unvisited
        unvisited.remove(best_n)

    return cycle, total_dist


def plot_graph(G: nx.Graph) -> None:
    """
    Plots a NetworkX graph with black edges and nodes as black dots with the node key on the outside.
    Nodes are plotted using their 'x' attribute (a tuple of x, y coordinates).

    Args:
        G (nx.Graph): The NetworkX graph to be plotted.
    """
    # Extract positions from the 'x' attribute of each node
    pos = {node: data['x'] for node, data in G.nodes(data=True)}

    # Draw nodes as black dots
    nx.draw_networkx_nodes(G, pos, node_color='green', node_size=10)

    # Create graph of all edges
    for edge in list(combinations(G.nodes, 2)):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='lightgrey', style='solid')

    # Draw edges in black
    offest_pos = {node: (x, y + -2) for node, (x,y) in pos.items()}
    nx.draw_networkx_edges(G, pos, edge_color='green', style='dashed')

    # Draw node labels on the outside
    nx.draw_networkx_labels(G, offest_pos, font_color='black', font_size=10, verticalalignment='top')


def generate_initial_population(
    G: nx.Graph,
    population_size: int
) -> Tuple[List[nx.Graph], List[float]]:
    """
    Generates an initial population of cycles for the TSP problem.

    Args:
        G (nx.Graph): The graph representing the TSP instance.
        population_size (int): The number of cycles to generate.

    Returns:
        List[nx.Graph]: A list of cycles, each represented as a NetworkX graph.
    """
    # Construct initial solution, cheapest insertion
    population = [None] * population_size
    lengths = [0] * population_size

    # Generate initial population
    for seed in range(population_size):
        tour = create_hamiltonian_tour(G=G, seed=seed)

        # Add to population
        population[seed] = tour[0]
        lengths[seed] = tour[1]

    return population, lengths


def fitness_probability(
    lengths: List[float],
    epsilon: float=10.,
) -> List[float]:
    """
    Calculate the probability of selection for each cycle in the population.

    Args:
        lengths (List[float]): The lengths of the cycles in the population.
        epsilon (float): A small value to avoid division by zero.

    Returns:
        List[float]: A list of cycles with their probabilities
    """
    # Get highest distance
    max_length = max(lengths) + epsilon

    # Sum of deviations from max distance
    total_dev = len(lengths) * max_length - sum(lengths)

    # Calculate probabilities
    probabilities = [(max_length - l) / total_dev for l in lengths]

    return probabilities


def select_parents(
    population: List[nx.Graph],
    probabilities: List[float],
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Selects two parents from the population using the roulette wheel selection method.

    Args:
        population (List[nx.Graph]): The population of cycles.
        probabilities (List[float]): The probabilities of selection for each cycle.

    Returns:
        Tuple[nx.Graph, nx.Graph]: The two selected parents.
    """
    # Select two parents
    return random.choices(population, weights=probabilities, k=2)


def get_viable_links_from_node(
    node: int,
    parent: nx.Graph,
    child: nx.Graph,
) -> List[Tuple[int, int, float]]:
    """
    Get the viable links from a node in the parent cycle that are not in the child cycle.

    Args:
        node (int): The node to consider.
        parent (nx.Graph): The parent cycle.
        child (nx.Graph): The child cycle.

    Returns:
        List[Tuple[int, int, float]]: A list of edges that are viable to be added to the child cycle.
    """
    return [e for e in parent.edges(node, data='d') if e[:2] not in child.edges]