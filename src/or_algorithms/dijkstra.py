import networkx as nx
from sortedcontainers import SortedDict
from typing import List, Optional, Dict, Tuple


class UnvisitedNodes:
    """Priority queue for unvisited nodes in the graph.

    This class maintains a sorted dictionary of nodes that need to be explored
    during shortest path calculations. Each entry stores:
        - The shortest known distance to the node.
        - The node's key.
        - The preceding node in the best-known path.

    Attributes:
        _queue (SortedDict[tuple[int, str, Optional[str]], None]):
            Internal queue storing unvisited nodes.
            Keys are (distance, node, preceding node).
    """

    def __init__(self) -> None:
        """Initializes an empty priority queue for unvisited nodes."""
        self._queue: SortedDict[Tuple[int, str, Optional[str]], None] = (
            SortedDict()
        )

    @property
    def size(self) -> int:
        """Returns the number of elements in the queue.

        Returns:
            int: The number of unvisited nodes.
        """
        return len(self._queue)

    @property
    def empty(self) -> bool:
        """Checks if the queue is empty.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return len(self._queue) == 0

    def insert(
        self, node_key: str, distance: int, preceding: Optional[str] = None
    ) -> None:
        """Inserts a node into the queue of unvisited nodes.

        Args:
            node_key (str): The key of the node.
            distance (int): The shortest known distance from the start node.
            preceding (Optional[str]): The preceding node in the path.

        Raises:
            TypeError: If `distance` is not an integer.
            ValueError: If `node_key` is empty or `distance` is negative.
        """
        if not isinstance(distance, int):
            raise TypeError("Distance must be an integer")

        if not node_key:
            raise ValueError("Node key must be a non-empty string")

        if distance < 0:
            raise ValueError("Distance cannot be negative")

        self._queue[(distance, node_key, preceding)] = None

    def peek(self) -> Tuple[int, str, Optional[str]]:
        """Retrieves the node with the shortest distance without removing it.

        Returns:
            Tuple[int, str, Optional[str]]: (distance, node key, preceding node)

        Raises:
            RuntimeError: If the queue is empty.
        """
        if self.empty:
            raise RuntimeError("Cannot peek at an empty queue")

        return next(iter(self._queue))

    def pop(self) -> Tuple[int, str, Optional[str]]:
        """Retrieves and removes the node with the shortest distance.

        Returns:
            Tuple[int, str, Optional[str]]: (distance, node key, preceding node)

        Raises:
            RuntimeError: If the queue is empty.
        """
        if self.empty:
            raise RuntimeError("Cannot pop from an empty queue")

        first_key = self.peek()
        self._queue.pop(first_key)

        return first_key

    def get_distance(self, target_key: str) -> int:
        """Finds a node in the queue by key and returns its distance.

        Args:
            target_key (str): The key of the target node.

        Returns:
            int: The shortest known distance to the node.

        Raises:
            KeyError: If the node key is not found in the queue.
        """
        for distance, node_key, _ in self._queue.keys():
            if node_key == target_key:
                return distance

        raise KeyError(f"Node key '{target_key}' not found in the queue")

    def get_element(self, target_key: str) -> Tuple[int, str, Optional[str]]:
        """Finds a node in the queue by key and returns its full entry.

        Args:
            target_key (str): The key of the target node.

        Returns:
            Tuple[int, str, Optional[str]]: (distance, node key, preceding node)

        Raises:
            KeyError: If the node key is not found in the queue.
        """
        for element in self._queue.keys():
            if element[1] == target_key:
                return element

        raise KeyError(f"Node key '{target_key}' not found in the queue")

    def includes(self, target_key: str) -> bool:
        """Checks if a node key exists in the queue.

        Args:
            target_key (str): The key of the target node.

        Returns:
            bool: True if the node key exists in the queue, False otherwise.
        """
        return any(
            node_key == target_key for _, node_key, _ in self._queue.keys()
        )

    def update_distance(
        self, target_key: str, new_distance: int, new_preceding: Optional[str]
    ) -> None:
        """Updates the distance of a node in the queue.

        Args:
            target_key (str): The key of the target node.
            new_distance (int): The new shortest distance.
            new_preceding (Optional[str]): The new preceding node.

        Raises:
            KeyError: If the node key is not found in the queue.
        """
        element = self.get_element(target_key)
        del self._queue[element]
        self.insert(target_key, new_distance, new_preceding)


def reconstruct_best_path(
    visited_nodes: Dict[str, Tuple[int, Optional[str]]], end_node: str
) -> List[str]:
    """Reconstructs the shortest path from the start node to the given end node.

    Args:
        visited_nodes (Dict[str, Tuple[int, Optional[str]]]):
            A dictionary mapping each node to its (distance, preceding node).
        end_node (str): The target node to reconstruct the path from.

    Returns:
        List[str]: The shortest path from start to end.

    Raises:
        KeyError: If `end_node` is not found in `visited_nodes`.
    """
    current_node: str | None = end_node
    shortest_path: List[str] = []

    while current_node:
        shortest_path.append(current_node)
        if current_node not in visited_nodes:
            raise KeyError(f"Node '{current_node}' not found in visited nodes")

        current_node = visited_nodes[current_node][1]

    return shortest_path[::-1]


def solve_shortest_path(
    graph: nx.Graph, start: str, end: str
) -> Tuple[int, List[str]]:
    """Finds the shortest path between two nodes in a weighted graph.

    Args:
        graph (nx.Graph): The input graph.
        start (str): The start node key.
        end (str): The end node key.

    Returns:
        Tuple[int, List[str]]: The shortest distance and the corresponding path.

    Raises:
        ValueError: If the start or end node is not in the graph.
    """
    if start not in graph:
        raise ValueError(f"Start node '{start}' not found in the graph")

    if end not in graph:
        raise ValueError(f"End node '{end}' not found in the graph")

    unvisited = UnvisitedNodes()
    visited = {}

    unvisited.insert(start, 0)

    while not unvisited.empty:
        curr_dist, curr_key, curr_previous = unvisited.pop()

        for nbr_key in graph.neighbors(curr_key):
            if nbr_key in visited:
                continue

            edge = graph.edges[curr_key, nbr_key]["weight"]
            new_distance = curr_dist + edge

            if unvisited.includes(nbr_key):
                if new_distance < unvisited.get_distance(nbr_key):
                    unvisited.update_distance(nbr_key, new_distance, curr_key)
            else:
                unvisited.insert(nbr_key, new_distance, curr_key)

        visited[curr_key] = (curr_dist, curr_previous)

    if end not in visited:
        raise ValueError(f"End node '{end}' is unreachable")

    return visited[end][0], reconstruct_best_path(visited, end)
