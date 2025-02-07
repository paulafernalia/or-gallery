import pytest
import networkx as nx
from or_algorithms import dijkstra as djks


@pytest.fixture
def short_queue():
    queue = djks.UnvisitedNodes()
    queue.insert("A", 10)
    queue.insert("B", 5)

    return queue


@pytest.fixture
def small_graph():
    # Create a graph
    graph = nx.Graph()

    # Define nodes and their attributes
    nodes = ["A", "B"]

    # Add nodes with attributes to the graph
    for node in nodes:
        graph.add_node(node)

    # Add edges with weights (distances)
    graph.add_weighted_edges_from([("A", "B", 4)])

    return graph


@pytest.fixture
def large_graph():
    # Create a graph
    graph = nx.Graph()

    # Define nodes and their attributes
    nodes = ["A", "B", "C", "D"]

    # Add nodes with attributes to the graph
    for node in nodes:
        graph.add_node(node)

    edges = [("A", "B", 1), ("A", "C", 3), ("B", "C", 1), ("C", "D", 1), ("B", "D", 4)]

    # Add edges with weights (distances)
    graph.add_weighted_edges_from(edges)

    return graph


def test_insert_and_peek_in_unvisitednodes(short_queue):
    assert short_queue.size == 2
    assert short_queue.peek() == (5, "B", None)


def test_pop_in_unvisitednodes(short_queue):
    popped1 = short_queue.pop()
    assert popped1 == (5, "B", None)

    popped2 = short_queue.pop()
    assert popped2 == (10, "A", None)


def test_get_distance(short_queue):
    match1 = short_queue.get_distance("B")
    assert match1 == 5

    match2 = short_queue.get_distance("A")
    assert match2 == 10


def test_update_distance(short_queue):
    short_queue.update_distance("A", 100, None)
    assert short_queue.peek() == (5, "B", None)

    short_queue.update_distance("A", 1, None)
    assert short_queue.peek() == (1, "A", None)


def test_shortest_path_small_graph(small_graph):
    # Find shortest path
    distance, path = djks.solve_shortest_path(small_graph, "A", "B")

    assert distance == 4
    assert path == ["A", "B"]


def test_shortest_path_large_graph(large_graph):
    # Find shortest path
    distance, path = djks.solve_shortest_path(large_graph, "A", "D")

    assert distance == 3
    assert path == ["A", "B", "C", "D"]
