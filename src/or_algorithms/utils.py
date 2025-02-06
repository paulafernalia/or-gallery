from typing import Dict
import math


def count_fractional_variables(solution: Dict[str, float]) -> int:
    """Count how many fractional variables are there in the solution"""

    if not solution:
        raise ValueError("Solution dictionary cannot be empty")

    count = 0

    for name, value in solution.items():
        if math.floor(value) != value:
            count += 1

    return count
