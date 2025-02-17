import pytest
import networkx as nx
from or_algorithms import kruskals as krk


def test_empty_tree():
    # Create a graph
    graph = nx.Graph()

    assert nx.is_isomorphic(krk.solve_minimum_spanning_tree(graph), graph)


def test_single_node_tree():
    # Create a graph
    graph = nx.Graph()
    graph.add_node("A")

    assert nx.is_isomorphic(krk.solve_minimum_spanning_tree(graph), graph)


def test_single_edge_tree():
    # Create a graph
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=3)

    assert nx.is_isomorphic(krk.solve_minimum_spanning_tree(graph), graph)


def test_double_node_tree():
    # Create a graph
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=3)
    graph.add_edge("B", "C", weight=2)

    assert nx.is_isomorphic(krk.solve_minimum_spanning_tree(graph), graph)


def test_mini_cycle_tree():
    # Create a graph
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=3)
    graph.add_edge("B", "C", weight=2)
    graph.add_edge("C", "A", weight=10)

    minsptree = nx.Graph()
    minsptree.add_edge("A", "B", weight=3)
    minsptree.add_edge("B", "C", weight=1)

    assert nx.is_isomorphic(krk.solve_minimum_spanning_tree(graph), minsptree)


def test_large_tree():
    # Create a graph
    graph = nx.Graph()
    minsptree = nx.Graph()
    
    nodes = ["A", "B", "C", "D", "E", "F", "G"]

    for node in nodes:
        graph.add_node(node)
        minsptree.add_node(node)

    edges = [
        ("A", "B", 7), ("B", "C", 8), ("C", "E", 5),
        ("E", "G", 9), ("G", "F", 11), ("F", "D", 6),
        ("D", "A", 5), ("D", "B", 9), ("B", "E", 7),
        ("E", "F", 8), ("D", "E", 15)
    ]

    graph.add_weighted_edges_from(edges)

    selected_edges = [
        ("A", "B", 7), ("C", "E", 5),
        ("E", "G", 9), ("F", "D", 6),
        ("D", "A", 5), ("B", "E", 7),
    ]

    minsptree.add_weighted_edges_from(selected_edges)

    assert nx.is_isomorphic(krk.solve_minimum_spanning_tree(graph), minsptree)
