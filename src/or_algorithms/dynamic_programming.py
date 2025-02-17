from typing import List, Tuple
import math


def total_value(
    W: int,
    i: int,
    ws: List[int],
    vs: List[float],
    hist: List[List[float]]
) -> float:
    """
    Recursively calculates the maximum total value that can be obtained
    with the given knapsack capacity and available items.

    Args:
        W (int): Remaining knapsack capacity.
        i (int): Current index of the item being considered.
        ws (List[int]): List of item weights.
        vs (List[int]): List of item values.
        hist (Dict[int, Dict[int, int]]): table to store computed results.

    Returns:
        int: The maximum total value achievable with the remaining capacity.
    """
    # If capacity is negative problem is infeasible
    if W < 0:
        return -math.inf

    # Check if this value was previously calculated
    value = hist[i][W]
    if value != -1:
        return value

    # Else use recurrent function to calculate best action
    hist[i][W] = max(
        vs[i] + total_value(W - ws[i], i + 1, ws, vs, hist),
        total_value(W, i + 1, ws, vs, hist)
    )

    return hist[i][W]


def backtrack(
    W: int,
    ws: List[int],
    vs: List[float],
    hist: List[List[float]]
) -> List[int]:
    """
    Determines the indices of the items to be included in the knapsack
    by backtracking through the memoization table.

    Args:
        W (int): Knapsack capacity.
        ws (List[int]): List of item weights.
        vs (List[int]): List of item values.
        hist (Dict[int, Dict[int, int]]): Memoization table with computed
                                          results.

    Returns:
        List[int]: List of selected item indices.
    """
    take = []

    for i in range(len(ws) + 1):
        if i < len(ws):
            if W >= ws[i]:
                # If taking item i leads to a better result than not
                if hist[i + 1][W - ws[i]] + vs[i] > hist[i + 1][W]:
                    take.append(i)
                    W -= ws[i]
        # else:

    return take


def solve_knapsack(
    W: int,
    ws: List[int],
    vs: List[float]
) -> Tuple[float, List[int]]:
    """
    Solves the 0/1 knapsack problem using dynamic programming.

    Args:
        W (int): Knapsack capacity.
        ws (List[int]): List of item weights.
        vs (List[int]): List of item values.

    Raises:
        IndexError: If the number of weights and values do not match.
        ValueError: If the knapsack capacity or any weight/value is negative.

    Returns:
        Tuple[int, List[int]]: The maximum value that can be obtained and
        the list of selected item indices.
    """
    if len(ws) != len(vs):
        raise IndexError("Weights and values must have the same length")

    if W < 0:
        raise ValueError("Knapsack capacity must be non negative")

    for i in range(len(ws)):
        if ws[i] < 0:
            raise ValueError("All weights must be positive")
        if vs[i] < 0:
            raise ValueError("All values must be positive")

    history = [
        [-1. for _ in range(W + 1)] for _ in range(len(vs) + 1)
    ]

    # Precompute some base cases
    for w in range(W + 1):
        history[len(vs)][w] = 0.

    for i in range(len(vs) + 1):
        history[i][0] = 0.

    # Maximise value
    value = total_value(W, 0, ws, vs, history)

    # Find which items to take
    policy = backtrack(W, ws, vs, history)

    return value, policy
