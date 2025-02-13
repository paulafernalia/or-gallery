from typing import List, Dict
import math


def solve_knapsack(
    W: int,
    ws: List[int],
    vs: List[int]
) -> int:

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
        [-1 for _ in range(W + 1)] for _ in range(len(vs) + 1)
    ]

    # Precompute some base cases
    for w in range(W + 1):
        history[len(vs)][w] = 0

    for i in range(len(vs) + 1):
        history[i][0] = 0

    return total_value(
        W, 0, ws, vs, history
    )


def total_value(
    W: int,
    i: int,
    ws: List[int],
    vs: List[int],
    hist: Dict[int, Dict[int, int]]
) -> int:
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
