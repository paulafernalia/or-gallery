import pytest
from or_algorithms import branch_and_bound as bb


@pytest.fixture(autouse=True)
def reset_node_count():
    """Automatically reset bb.Node.count before every test"""
    bb.Node.count = 0


def test_node_constructor():
    """Test creating nodes"""
    node1 = bb.Node()

    assert node1.key == 1
    assert node1.depth == 1
    assert bb.Node.count == 1

    node2 = bb.Node()

    assert node2.key == 2
    assert node2.depth == 1
    assert bb.Node.count == 2


def test_node_branch():
    """Test branching on a node"""
    node = bb.Node()
    node.branch()

    assert node.left and node.left.parent == node
    assert node.right and node.right.parent == node
    assert node.left.key == 2
    assert node.right.key == 3


@pytest.fixture
def simple_tree():
    """Fixture for a simple tree with a root node"""
    tree = bb.Tree()
    return tree


@pytest.fixture
def big_tree():
    """Fixture for a larger tree"""
    tree = bb.Tree()
    tree.root.branch()
    tree.root.left.branch()
    return tree


def test_search_simple_tree(simple_tree):
    """Test searching for a node in a tree with only a root node"""
    assert simple_tree.search(1, simple_tree.root) == simple_tree.root
    assert not simple_tree.search(6, simple_tree.root)


def test_search_larger_tree(big_tree):
    """Test searching for a node in a larger tree"""
    assert big_tree.search(5, big_tree.root) == big_tree.root.left.right
    assert big_tree.search(2, big_tree.root) == big_tree.root.left
    assert big_tree.search(4, big_tree.root) == big_tree.root.left.left
    assert big_tree.search(1, big_tree.root) == big_tree.root
    assert not big_tree.search(6, big_tree.root)


def test_get_path_to_root_simple_tree(simple_tree):
    """Test getting path to root in a tree with only a root node"""
    assert simple_tree.get_path_to_root(simple_tree.root) == []


def test_get_path_to_root_big_tree(big_tree):
    """Test getting path to root in a tree with only a root node"""
    expected_path = [big_tree.root.left, big_tree.root]
    assert big_tree.get_path_to_root(big_tree.root.left.left) == expected_path


def test_find_intersection_simple_tree(simple_tree):
    assert not simple_tree.find_intersection(
        simple_tree.root, simple_tree.root
    )


def test_find_intersection_big_tree(big_tree):
    obtained = big_tree.find_intersection(
        big_tree.root.left.left, big_tree.root.left.right
    )

    assert obtained == big_tree.root.left


def test_find_intersection_super_big_tree(big_tree):
    # Create nodes 6 and 7
    big_tree.root.left.right.branch()

    obtained = big_tree.find_intersection(
        big_tree.root.left.left, big_tree.root.left.right.left
    )

    assert obtained == big_tree.root.left


@pytest.fixture
def empty_unexplored_list():
    """Fixture for a simple tree with a root node"""
    unexplored_list = bb.UnexploredList()
    return unexplored_list


def test_insert_in_unexplored_list(empty_unexplored_list):
    """Test inserting a node in the queue"""
    node = bb.Node(obj=10)
    empty_unexplored_list.insert(node)

    assert not empty_unexplored_list.is_empty
    assert empty_unexplored_list.size == 1

    assert empty_unexplored_list.peek() == node
    assert empty_unexplored_list.size == 1
    assert not empty_unexplored_list.is_empty

    assert empty_unexplored_list.pop() == node
    assert empty_unexplored_list.size == 0
    assert empty_unexplored_list.is_empty
