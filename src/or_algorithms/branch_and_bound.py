import heapq
import pulp


class VariableBound:
    """
    Represents a variable bound in a branch-and-bound algorithm.
    
    Attributes:
        varName (str | None): Name of the variable.
        direction (str): 'U' for upper bound, 'L' for lower bound.
        bound (int): The bound value.
    """
    def __init__(self, varName=None, direction=None, bound=None):
        assert direction in ('U', 'L')
        assert isinstance(bound, int)

        self.varName = varName
        self.direction = direction
        self.bound = bound


class Node:
    """
    Represents a node in a branch-and-bound tree.
    
    Attributes:
        key (int): Unique identifier of the node.
        left (Node | None): Left child node.
        right (Node | None): Right child node.
        parent (Node | None): Parent node.
        objective (float | None): Solution to the LP relaxation of the node.
        is_integer (bool): Whether the solution is integer.
        depth (int): Depth within the branch-and-bound tree.
        var_bound (VariableBound): The variable bound associated with this node.
    """
    count = 0

    def __init__(self, parent=None, objective=None, left=None, right=None):
        Node.count += 1
        self.key = Node.count
        self.left = left
        self.right = right
        self.parent = parent
        self.objective = objective
        self.is_integer = False
        self.var_bound = VariableBound()
        self.depth = parent.depth + 1 if parent else 1

    def branch(self):
        """Creates two children from a node."""
        assert not self.left and not self.right
        self.left = Node(parent=self)
        self.right = Node(parent=self)

    def __repr__(self):
        """Returns a string representation of the node."""
        return f"Node({self.key})"


class Tree:
    """
    Represents a binary tree for the branch-and-bound process.
    
    Attributes:
        root (Node): Root node of the tree.
        node_limit (int): Maximum number of nodes allowed.
    """
    def __init__(self):
        self.root = Node()
        self.node_limit = int(1e6)

    def search(self, key, node):
        """
        Finds a node in the tree that matches the given key.
        
        Args:
            key (int): The unique ID of the node to find.
            node (Node | None): The node from which to start searching.

        Returns:
            Node | None: The matching node if found, else None.
        """
        assert key > 1

        if not node or node.key > key:
            return None

        if node.key == key:
            return node

        return self.search(key, node.left) or self.search(key, node.right)

    def get_path_to_root(self, node):
        """Returns the path from the given node to the root."""
        assert node
        path = []
        while node.parent:
            path.append(node.parent)
            node = node.parent
        return path

    def find_intersection(self, node1, node2):
        """Finds the common ancestor of two nodes."""
        assert node1 and node2

        path1 = self.get_path_to_root(node1)
        path2 = self.get_path_to_root(node2)

        path1_set = set(path1)
        for node in path2:
            if node in path1_set:
                return node

        return None


class UnexploredList:
    """Priority queue for unexplored nodes, sorted by objective value."""
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join(str(node.key) for _, _, node in self.queue)

    def is_empty(self):
        """Checks if the queue is empty."""
        return len(self.queue) == 0

    def insert(self, node):
        """Inserts a node into the priority queue."""
        heapq.heappush(self.queue, (-node.objective, node.key, node))

    def pop(self):
        """Removes and returns the node with the highest priority."""
        if self.is_empty():
            raise IndexError("pop from an empty queue")
        return heapq.heappop(self.queue)[2]
    
    def peek(self):
        """Returns the node with the highest priority without removing it."""
        return self.queue[0][2] if not self.is_empty() else None

    def size(self):
        """Returns the number of nodes in the queue."""
        return len(self.queue)


class Bounds:
    """Lower and upper bounds for branch and bound algorithm"""
    def __init__(self):
        self.best_objective = 1e10
        self.best_bound = -1e10
        self.absolute_tol = 1
        self.relative_tol = 1e-5

    def check_convergence(self):
        """Function to check if the algorithm has converged"""
        if (self.best_objective - self.best_bound) / self.best_bound < self.relative_tol:
            return True

        if (self.best_objective - self.best_bound) < self.absolute_tol:
            return True

        return False



class BranchAndBound:
    """Branch-and-bound solver using a priority queue."""
    def __init__(self, model):
        self.tree = Tree()
        self.unexplored_list = UnexploredList()
        self.bounds = Bounds()
        self.model = model
        self.best_solution = None
        self.node_limit = 1e5

    def converges(self):
        """Checks if the algorithm should stop based on convergence criteria."""
        if self.unexplored_list.is_empty():
            return True

        # If maximum number of nodes is exceeded
        if Node.count > self.node_limit:
            return True

        # If optimality gap is below tolerances
        if self.bounds.check_convergence():
            return True

        # If best unexplored node is worse than best integer solution
        next_best_node = self.unexplored_list.peek()
        if next_best_node and next_best_node.objective > self.best_objective:
            self.unexplored_list = UnexploredList()
            return True

        return False

    def is_solution_integer(self):
        """Checks if the solution is integer-valued."""
        return all(var.value().is_integer() for var in self.model.variables())

    def solve_node(self, node):
        """Solves the LP relaxation at a given node."""
        self.model.solve()
        node.objective = self.model.objective.value()

        if node.objective > self.best_objective:
            return

        if self.is_solution_integer():
            node.is_integer = True
            self.best_objective = node.objective
            self.best_solution = None  # TODO: Update best solution
        else:
            self.unexplored_list.insert(node)
            self.best_bound = self.unexplored_list.peek().objective

    def solve(self):
        """Runs the branch-and-bound algorithm."""
        self.solve_node(self.tree.root)
        previous_node = self.tree.root

        while not self.unexplored_list.is_empty():
            selected_node = self.unexplored_list.pop()
            assert selected_node.objective < self.best_objective

            self.solve_node(selected_node)
            if self.converges():
                return
            previous_node = selected_node


