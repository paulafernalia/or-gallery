import heapq
import pulp
import math


class VariableBound:
    """
    Represents a variable bound in a branch-and-bound algorithm.

    Attributes:
        varName (str | None): Name of the variable.
        dir_ (str): 'U' for upper bound, 'L' for lower bound.
        bound (int): The bound value.
    """
    def __init__(self, varName=None, dir_=None, bound=None):
        assert not dir_ or dir_ in ('U', 'L')
        assert not bound or isinstance(bound, int)

        self.varName = varName
        self.dir_ = dir_
        self.bound = bound


class Node:
    """
    Represents a node in a branch-and-bound tree.

    Attributes:
        key (int): Unique identifier of the node.
        left (Node | None): Left child node.
        right (Node | None): Right child node.
        parent (Node | None): Parent node.
        obj (float | None): Solution to the LP relaxation of the node.
        is_integer (bool): Whether the solution is integer.
        depth (int): Depth within the branch-and-bound tree.
        var_bound (VariableBound): The variable bound linked to this node.
    """
    count = 0

    def __init__(self, parent=None, obj=None, left=None, right=None):
        Node.count += 1
        self.key = Node.count
        self.left = left
        self.right = right
        self.parent = parent
        self.obj = obj
        self.is_integer = False
        self.bound = VariableBound()
        self.previous_bound = VariableBound()
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
        assert key >= 1

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
        self._queue = []

    def __str__(self):
        return ' '.join(str(node.key) for _, _, node in self._queue)

    @property
    def is_empty(self):
        """Checks if the queue is empty."""
        return len(self._queue) == 0

    def insert(self, node):
        """Inserts a node into the priority queue."""
        assert node and node.obj
        heapq.heappush(self._queue, (node.obj, node.key, node))

    def pop(self):
        """Removes and returns the node with the highest priority."""
        if self.is_empty:
            raise IndexError("pop from an empty queue")
        return heapq.heappop(self._queue)[2]

    def peek(self):
        """Returns the node with the highest priority without removing it."""
        return self._queue[0][2] if not self.is_empty else None

    @property
    def size(self):
        """Returns the number of nodes in the queue."""
        return len(self._queue)


class Bounds:
    """Lower and upper bounds for branch and bound algorithm"""
    def __init__(self):
        self._best_obj = 1e10
        self._best_bound = -1e10
        self._absolute_tol = 1
        self._relative_tol = 1e-5
        self._absolute_gap = 1e10
        self._relative_gap = 1e10

    def update_gaps(self):
        self._absolute_gap = self._best_obj - self._best_bound
        self._relative_gap = abs(self._absolute_gap / self._best_bound)

    def update_best_obj(self, value):
        assert value >= self._best_bound
        assert value < self._best_obj
        self._best_obj = value
        self.update_gaps()

    def update_best_bound(self, value):
        assert self._best_obj >= value
        assert value > self._best_bound
        self._best_bound = value
        self.update_gaps()

    @property
    def best_obj(self):
        return self._best_obj

    @property
    def best_bound(self):
        return self._best_bound

    def check_convergence(self):
        """Function to check if the algorithm has converged"""
        if self._absolute_gap < self._absolute_tol:
            return True

        if self._relative_gap < self._relative_tol:
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
        """Checks if the algorithm shas converged."""
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
        if next_best_node and next_best_node.obj > self.bounds.best_obj:
            self.unexplored_list = UnexploredList()
            return True

        return False

    def is_solution_integer(self):
        """Checks if the solution is integer-valued."""
        return all(var.value().is_integer() for var in self.model.variables())

    def reset_model(self, origin_node, target_node):
        if origin_node == target_node or not origin_node.parent:
            return

        # Remove variable bounds
        if not origin_node.bound:
            return

        # Get a variable by name
        variables_dict = self.model.variablesDict()

        print(origin_node.bound.varName, origin_node.key)
        var = variables_dict[origin_node.bound.varName]

        assert origin_node.bound.varName == origin_node.previous_bound.varName
        assert origin_node.bound.dir == origin_node.previous_bound.dir

        if origin_node.bound.dir == 'U':
            var.upBound = origin_node.previous_bound.bound
        else:
            var.lowBound = origin_node.previous_bound.bound

        assert origin_node.parent

        self.reset_model(origin_node.parent)

    def update_model(self, origin_node, target_node):
        if origin_node == target_node:
            return

        assert target_node.parent

        # Backtrack to start from the origin node
        self.update_model(origin_node, target_node.parent)

        # Add variable bounds
        variables_dict = self.model.variablesDict()
        var = variables_dict[origin_node.bounds.varName]

        assert origin_node.bound.varName == origin_node.previous_bound.varName
        assert origin_node.bound.dir == origin_node.previous_bound.dir

        if origin_node.bounds.dir == 'U':
            var.upBound = origin_node.bound.bound
        else:
            var.lowBound = origin_node.bound.bound

    def traverse(self, origin, destination):
        common_node = self.tree.find_intersection(origin, destination)

        # Reset model to common node (remove variable bounds up branch)
        self.reset_model(origin, common_node)

        # Add variable bounds down branch
        self.update_model(common_node, destination)

    def find_most_fractional_variable(self):
        # TODO(paula): Check solution is feasible with assertion
        best_fraction = 0
        best_var = None

        for var in self.model.variables():
            lower = var.value() - math.floor(var.value())
            upper = math.ceil(var.value()) - var.value()

            fraction = max(lower, upper)

            if fraction < 1 and fraction > best_fraction:
                best_fraction = fraction
                best_var = var

        assert best_fraction > 0

        return best_var

    def branch(self, node):
        # Create branches
        node.branch()

        # Select most fractional variable in solution
        var = self.find_most_fractional_variable()

        # Add variable bound to children nodes
        node.left.bound = VariableBound(
            var.name,  'U', math.floor(var.value())
        )
        node.left.previous_bound = VariableBound(var.name,  'U', var.upBound)

        node.right.bound = VariableBound(
            var.name, 'L', math.ceil(var.value())
        )
        node.left.previous_bound = VariableBound(var.name,  'L', var.lowBound)

    def solve_node(self, node):
        """Solves the LP relaxation at a given node."""
        self.model.solve(pulp.PULP_CBC_CMD(msg=False))
        node.obj = self.model.obj.value()

        print()
        print(f"SOLVE NODE {node.key}, obj={node.obj}")

        if node.obj > self.bounds.best_obj:
            return

        if self.is_solution_integer():
            node.is_integer = True
            self.best_obj = node.obj
            self.best_solution = None  # TODO: Update best solution
        else:
            self.unexplored_list.insert(node)
            self.best_bound = self.unexplored_list.peek().obj

    def optimize(self):
        """Runs the branch-and-bound algorithm."""
        # Solve root node
        self.solve_node(self.tree.root)
        previous_node = self.tree.root

        # If list of unexplored is not empty, branch
        while not self.unexplored_list.is_empty():
            # Look for next node to solve (best bound search)
            selected_node = self.unexplored_list.pop()
            assert selected_node.obj < self.bounds.best_obj

            # Update model to the selected_node
            self.traverse(previous_node, selected_node)

            # Branch on selected node
            self.branch(selected_node)

            # Solve left child node
            self.traverse(selected_node, selected_node.left)
            self.solve_node(selected_node.left)

            # Solve right child node
            self.traverse(selected_node.left, selected_node.right)
            self.solve_node(selected_node.right)

            # Check if we found the optimal solution or proved infeasibility
            if self.converges():
                return

            # Make current node the previos node
            previous_node = selected_node
