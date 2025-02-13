from or_algorithms import dynamic_programming as dp


def test_empty_knapsack():
    capacity = 0
    weights = [1, 2]
    values = [3, 4]

    value, policy = dp.solve_knapsack(capacity, weights, values)

    assert value == 0
    assert policy == []


def test_empty_list_of_items():
    capacity = 10
    weights = []
    values = []

    value, policy = dp.solve_knapsack(capacity, weights, values)

    assert value == 0
    assert policy == []


def test_simple_knapsack_fits():
    capacity = 10
    weights = [8]
    values = [3]

    value, policy = dp.solve_knapsack(capacity, weights, values)

    assert value == 3
    assert policy == [0]


def test_simple_knapsack_no_fit():
    capacity = 8
    weights = [10]
    values = [3]

    value, policy = dp.solve_knapsack(capacity, weights, values)

    assert value == 0
    assert policy == []


def test_simple_knapsack_three_items():
    capacity = 10
    weights = [10, 4, 6]
    values = [3, 2, 2]

    value, policy = dp.solve_knapsack(capacity, weights, values)

    assert value == 4
    assert policy == [1, 2]
