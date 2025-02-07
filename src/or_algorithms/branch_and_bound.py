import heapq
import pulp
import math
from typing import Dict, List, Optional
from or_algorithms import utils

BB_UNBOUNDED = -2
BB_INFEASIBLE = -1
BB_NOT_CONVERGED = 0
BB_OPTIMAL = 1
BB_NODE_LIMIT = 2


class VariableBound:
    """Represents a variable bound in a branch-and-bound algorithm.

    Attributes:
        varName (str | None): Name of the variable.
        dir_ (str): 'U' for upper bound, 'L' for lower bound.
        bound (int): The bound value.
    """

    def __init__(
        self,
        varName: str | None = None,
        dir_: str | None = None,
        bound: int | None = None,
    ):
        """Initializes a VariableBound instance.

        Args:
            varName (str | None): The name of the variable. Defaults to None.
            dir_ (str | None): The direction of the bound, either 'U' (upper) or 'L' (lower). Defaults to None.
            bound (int | None): The numerical bound value. Defaults to None.

        Raises:
            ValueError: If `dir_` is not 'U' or 'L' when provided.
            TypeError: If `bound` is not an integer or None.
        """
        if dir_ is not None and dir_ not in {"U", "L"}:
            raise ValueError("dir_ must be 'U' (upper) or 'L' (lower).")

        if bound is not None and not isinstance(bound, int):
            raise TypeError("bound must be an integer or None.")

        self.varName = varName
        self.dir_ = dir_
        self.bound = bound

    def __repr__(self) -> str:
        """Returns a string representation of the VariableBound instance.

        Returns:
            str: A formatted string representation of the object.
        """
        return (
            f"VariableBound(varName={self.varName!r}, dir_={self.dir_!r}, "
            f"bound={self.bound!r})"
        )


class Node:
    """Represents a node in a branch-and-bound tree.

    Attributes:
        key (int): Unique identifier of the node.
        left (Node | None): Left child node.
        right (Node | None): Right child node.
        parent (Node | None): Parent node.
        obj (float | None): Solution to the LR of the node.
        depth (int): Depth within the branch-and-bound tree.
        var_bound (VariableBound | None): The variable bound linked to this
            node.
        solution (Optional[Dict[str, float]]): Solution to the LR of the node.
    """

    count = 0

    def __init__(
        self, parent=None, left=None, right=None, obj=math.inf
    ) -> None:
        """Initialises a new node.

        Args:
            parent (Node | None): Parent node.
            obj (float): Solution to the LR of the node.
            left (Node | None): Left child node.
            right (Node | None): Right child node.
        """
        Node.count += 1
        self.key: int = Node.count
        self.left: Optional[Node] = left
        self.right: Optional[Node] = right
        self.parent: Optional[Node] = parent
        self.obj: float = obj
        self.bound: VariableBound = VariableBound()
        self.previous_bound: VariableBound = VariableBound()
        self.depth: int = parent.depth + 1 if parent else 1
        self.solution: Optional[Dict[str, float]] = None

    def branch(self) -> None:
        """Creates two children from a node."""
        if self.left or self.right:
            raise RuntimeError("Cannot branch: Node already has children.")

        self.left = Node(parent=self)
        self.right = Node(parent=self)

    def __repr__(self) -> str:
        """Returns a string representation of the node.

        Returns:
            str: String representation of the node.
        """
        return (
            f"Node({self.key}), obj={self.obj}, bound={self.bound}, "
            f"previous_bound={self.previous_bound}"
        )


class Tree:
    """Represents a binary tree for the branch-and-bound process.

    Attributes:
        root (Node): Root node of the tree.
        node_limit (int): Maximum number of nodes allowed.
    """

    def __init__(self) -> None:
        """Initializes a new Tree with a root node and a node limit.

        Args:
            None
        """
        self.root: Node = Node()
        self.node_limit: int = int(1e6)

    def search(self, key: int, node: Node | None) -> Node | None:
        """Finds a node in the tree that matches the given key.

        Args:
            key (int): The unique ID of the node to find.
            node (Node | None): The node from which to start
                searching.

        Returns:
            Node | None: The matching node if found, else None.

        Raises:
            ValueError: If the key is less than 1.
        """
        if key < 1:
            raise ValueError("Key must be greater than or equal to 1")

        if not node or node.key > key:
            return None

        if node.key == key:
            return node

        return self.search(key, node.left) or self.search(key, node.right)

    def get_path_to_root(self, node: Node) -> list[Node]:
        """Returns the path from the given node to the root.

        Args:
            node (Node): The node from which to start the path.

        Returns:
            list[Node]: List of nodes from the given node to the root.

        Raises:
            ValueError: If the node is None.
        """
        if not node:
            raise ValueError("Node cannot be None")

        path: list[Node] = [node]
        while node.parent:
            path.append(node.parent)
            node = node.parent
        return path

    def find_intersection(self, node1: Node, node2: Node) -> Node | None:
        """Finds the common ancestor of two nodes.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            Node | None: The common ancestor node if found, else None.

        Raises:
            ValueError: If either node1 or node2 is None.
        """
        if not node1 or not node2:
            raise ValueError("Both nodes must be provided")

        path1 = self.get_path_to_root(node1)
        path2 = self.get_path_to_root(node2)

        path1_set = set(path1)
        for node in path2:
            if node in path1_set:
                return node

        return None


class UnexploredList:
    """Priority queue for unexplored nodes, sorted by objective value.

    Attributes:
        _queue (List[tuple]): Internal list used as a priority queue.
    """

    def __init__(self) -> None:
        """Initializes an empty priority queue."""
        self._queue: List[tuple] = []

    def __str__(self) -> str:
        """Returns a string representation of the queue."""
        return " ".join(str(node.key) for _, _, node in self._queue)

    @property
    def is_empty(self) -> bool:
        """Checks if the queue is empty.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return len(self._queue) == 0

    def insert(self, node: "Node") -> None:
        """Inserts a node into the priority queue.

        Args:
            node (Node): The node to insert.

        Raises:
            ValueError: If the node or its objective value is None.
        """
        if not node or node.obj is None:
            raise ValueError("Node and its objective value must be provided")
        heapq.heappush(self._queue, (node.obj, node.key, node))

    def pop(self) -> "Node":
        """Removes and returns the node with the highest priority.

        Returns:
            Node: The node with the highest priority.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty:
            raise IndexError("pop from an empty queue")
        return heapq.heappop(self._queue)[2]

    def peek(self) -> "Node":
        """Returns the node with the highest priority without removing it.

        Returns:
            Optional[Node]: The node with the highest priority, or None if
                the queue is empty.
        """
        if self.is_empty:
            raise ValueError("Cannot take a peek at an empty list")

        return self._queue[0][2]

    @property
    def size(self) -> int:
        """Returns the number of nodes in the queue.

        Returns:
            int: The number of nodes in the queue.
        """
        return len(self._queue)


class Bounds:
    """Lower and upper bounds for the branch-and-bound algorithm.

    Attributes:
        _best_obj (float): Best objective value found.
        _best_bound (float): Best bound value found.
        _absolute_tol (float): Absolute tolerance for convergence.
        _relative_tol (float): Relative tolerance for convergence.
        _absolute_gap (float): Absolute gap between best objective and
            best bound.
        _relative_gap (float): Relative gap between best objective and
            best bound.
    """

    def __init__(self) -> None:
        """Initializes the bounds with default values."""
        self._best_obj: float = math.inf
        self._best_bound: float = -math.inf
        self._absolute_tol: float = 1
        self._relative_tol: float = 1e-5
        self._absolute_gap: float = math.inf
        self._relative_gap: float = math.inf

    def update_gaps(self) -> None:
        """Updates the absolute and relative gaps."""
        self._absolute_gap = self._best_obj - self._best_bound
        self._relative_gap = abs(self._absolute_gap / self._best_bound)

    def update_best_obj(self, value: float) -> None:
        """Updates the best objective value found.

        Args:
            value (float): New best objective value.

        Raises:
            ValueError: If the value is not within the valid range.
        """
        if value >= self._best_obj:
            raise ValueError(
                f"New objective {value} cannot be worse than current best {self._best_bound}"
            )

        self._best_obj = value
        self.update_gaps()

    def update_best_bound(self, value: float) -> None:
        """Updates the best bound value found.

        Args:
            value (float): New best bound value.

        Raises:
            ValueError: If the value is not within the valid range.
        """
        if self._best_obj < value:
            raise ValueError(
                f"New bound {value} cannot be worse than best objective {self._best_obj}"
            )

        if value < self._best_bound:
            raise ValueError(
                f"New bound {value} cannot be better than current best {self._best_bound}"
            )

        self._best_bound = value
        self.update_gaps()

    @property
    def best_obj(self) -> float:
        """Returns the best objective value found.

        Returns:
            float: The best objective value.
        """
        return self._best_obj

    @property
    def best_bound(self) -> float:
        """Returns the best bound value found.

        Returns:
            float: The best bound value.
        """
        return self._best_bound

    @property
    def gap(self) -> float:
        """Returns the relative optimality gap.

        Returns:
            float: The relative optimality gap.
        """
        return self._relative_gap

    def check_convergence(self) -> bool:
        """Checks if the algorithm has converged.

        Returns:
            bool: True if the algorithm has converged, False otherwise.
        """
        if self._absolute_gap < self._absolute_tol:
            return True

        if self._relative_gap < self._relative_tol:
            return True

        return False


class BranchAndBound:
    """Branch-and-bound solver using a priority queue.

    Attributes:
        tree (Tree): The search tree for the algorithm.
        unexplored_list (UnexploredList): Priority queue for unexplored
            nodes.
        bounds (Bounds): Bounds for the branch-and-bound algorithm.
        model (pulp.LpProblem): The optimization model.
        solution (Optional[Dict[str, float]]): The optimal solution
            found.
        node_limit (int): Maximum number of nodes to explore.
        z (float): The best objective value found.
    """

    def __init__(self, model: pulp.LpProblem) -> None:
        """Initializes the Branch-and-Bound solver.

        Args:
            model (pulp.LpProblem): The optimization model.
        """
        self.tree = Tree()
        self.unexplored_list = UnexploredList()
        self.bounds = Bounds()
        self.model = model
        self.solution: Optional[Dict[str, float]] = None
        self.node_limit = int(1e5)
        self.z = 1e10

    def check_convergence(self) -> int:
        """Checks if the algorithm has converged.

        Returns:
            str: The convergence status.

        Raises:
            ValueError: If the solution is not found and unexplored
                list is empty.
        """

        # If list of unexplored nodes is empty
        if self.unexplored_list.is_empty:
            # Update best bound
            self.bounds.update_best_bound(self.bounds.best_obj)

            # If a solution was found, problem is optimal
            if self.solution:
                # Mark as optimal
                return BB_OPTIMAL

            # Else the problem is infeasible
            return BB_INFEASIBLE

        # If optimality gap is below tolerances and list is not empty
        if self.bounds.check_convergence():
            return BB_OPTIMAL

        # If maximum number of nodes is exceeded
        if Node.count > self.node_limit:
            return BB_NODE_LIMIT

        # If best unexplored node is worse than best integer solution
        next_best_node = self.unexplored_list.peek()
        if next_best_node and next_best_node.obj > self.bounds.best_obj:
            self.unexplored_list = UnexploredList()
            if self.solution:
                raise ValueError("Solution cannot exist")
            return BB_INFEASIBLE

        return BB_NOT_CONVERGED

    def is_solution_integer(self) -> bool:
        """Checks if the solution is integer-valued.

        Returns:
            bool: True if the solution is integer-valued, False
                otherwise.
        """
        return all(var.value().is_integer() for var in self.model.variables())

    def reset_model(self, origin_node: Node, target_node: Node) -> None:
        """Resets the model from the origin node to the target node.

        Args:
            origin_node (Node): The origin node.
            target_node (Node): The target node.
        """
        # Remove variable bounds
        if origin_node.bound.varName:
            # Get a variable by name
            variables_dict = self.model.variablesDict()
            var = variables_dict[origin_node.bound.varName]

            if origin_node.bound.varName != origin_node.previous_bound.varName:
                raise ValueError("Bound variable names do not match")
            if origin_node.bound.dir_ != origin_node.previous_bound.dir_:
                raise ValueError("Bound directions do not match")

            if origin_node.bound.dir_ == "U":
                var.upBound = origin_node.previous_bound.bound
            else:
                var.lowBound = origin_node.previous_bound.bound

        if origin_node == target_node:
            return

        if not origin_node.parent:
            raise ValueError("Origin node has no parent")
        self.reset_model(origin_node.parent, target_node)

    def update_model(self, origin_node: Node, target_node: Node) -> None:
        """Updates the model from the origin node to the target node.

        Args:
            origin_node (Node): The origin node.
            target_node (Node): The target node.
        """
        # Backtrack to start from the origin node
        if origin_node != target_node:
            if not target_node.parent:
                raise ValueError("Target node has no parent")
            self.update_model(origin_node, target_node.parent)

        if target_node.bound.varName:
            # Add variable bounds
            variables_dict = self.model.variablesDict()
            var = variables_dict[target_node.bound.varName]

            if target_node.bound.varName != target_node.previous_bound.varName:
                raise ValueError("Bound variable names do not match")
            if target_node.bound.dir_ != target_node.previous_bound.dir_:
                raise ValueError("Bound directions do not match")

            if target_node.bound.dir_ == "U":
                var.upBound = target_node.bound.bound
            else:
                var.lowBound = target_node.bound.bound

    def traverse(self, origin: Node, destination: Node) -> None:
        """Traverses from the origin node to the destination node.

        Args:
            origin (Node): The origin node.
            destination (Node): The destination node.
        """
        common_node = self.tree.find_intersection(origin, destination)

        if not common_node:
            raise ValueError("Intersection between paths must exist")

        # Reset model to common node (remove variable bounds up branch)
        self.reset_model(origin, common_node)

        # Add variable bounds down branch
        self.update_model(common_node, destination)

    def find_most_fractional_variable(self) -> pulp.LpVariable:
        """Finds the most fractional variable in the solution.

        Returns:
            pulp.LpVariable: The most fractional variable.

        Raises:
            ValueError: If no fractional variable is found.
        """
        best_fraction = 0
        best_var = None

        for var in self.model.variables():
            lower = var.value() - math.floor(var.value())
            upper = math.ceil(var.value()) - var.value()

            fraction = max(lower, upper)

            if 0 < fraction < 1 and fraction > best_fraction:
                best_fraction = fraction
                best_var = var

        if best_fraction <= 0:
            raise ValueError("No fractional variable found")

        return best_var

    def branch(self, node: Node) -> None:
        """Creates branches from the given node.

        Args:
            node (Node): The node to branch from.
        """
        # Create branches
        node.branch()

        # Reoptimise to get previous solution to the node model
        self.model.solve(pulp.PULP_CBC_CMD(msg=False))
        if self.model.objective.value() != node.obj:
            raise ValueError("Objective value mismatch")

        # Select most fractional variable in solution
        var = self.find_most_fractional_variable()

        assert node.left and node.right

        # Add variable bound to children nodes
        node.left.bound = VariableBound(var.name, "U", math.floor(var.value()))
        node.left.previous_bound = VariableBound(var.name, "U", var.upBound)

        node.right.bound = VariableBound(var.name, "L", math.ceil(var.value()))
        node.right.previous_bound = VariableBound(var.name, "L", var.lowBound)

    def print_headers(self) -> None:
        """Print headers of branch and bound progress."""
        print("Node  | Unexpl |        Obj | IntInf |")
        print(" LowBound  | UpperBound |       Gap")
        print("------|--------|------------|--------|")
        print("-----------|------------|-----------")

    def print_inner_node_progress(self, node: Node) -> None:
        if not node.parent:
            raise ValueError("Inner node must have a parent")

        if -math.inf < node.obj < math.inf:
            frac_vars = utils.count_fractional_variables(node.solution)
        else:
            frac_vars = 0

        """Print current progress to screen with aligned columns."""
        output = f" {node.key:<5}|"  # Left-aligned, width 6
        output += f" {self.unexplored_list.size:>6} |"

        if node.obj > self.bounds.best_obj:
            output += "     cutoff |"
        else:
            output += f" {node.obj:>10,.2f} |"

        output += f" {frac_vars:>6} |"

        bound_mod = min(self.bounds.best_bound, node.parent.obj)
        output += f" {bound_mod:>9,.2f} |"

        if self.solution:
            output += f" {self.bounds.best_obj:>10,.2f} |"

            gap_mod = (self.bounds.best_obj - bound_mod) / bound_mod
            output += f" {gap_mod * 100:>8,.2f}% |"
        else:
            output += "          - |         - |"

        print(output)

    def print_root_node_progress(self) -> None:
        if self.tree.root.obj == math.inf:
            print("Root node LP infeasible")
        elif self.tree.root.obj == -math.inf:
            print("Root node solved. Objective=inf")
        else:
            frac_vars = utils.count_fractional_variables(
                self.tree.root.solution
            )

            output = "Root node LP solved. "
            output += f"Objective={self.tree.root.obj:,.2f}. "
            output += f"Fractional variables={frac_vars}.\n"

            print(output)

    def print_mip_solution(self) -> None:
        # Print solution of each variable to screen
        if self.solution:
            output = "\nMIP solved. "
            output += f"Best objective={self.bounds.best_obj:,.4f}. "
            output += f"Best bound={self.bounds.best_bound:,.4f}. "
            output += f"Gap={self.bounds.gap * 100:,.4f}%."
            print(output)

            print("\nSolution:")
            for name, value in self.solution.items():
                print(f"* {name} = {value:,.4f}")

            assert utils.count_fractional_variables(self.solution) == 0
        else:
            print("\nProblem infeasible.")

    def solve_node(self, node: Node) -> None:
        """Solves the LP relaxation at a given node.

        Args:
            node (Node): The node to solve.
        """
        status = self.model.solve(pulp.PULP_CBC_CMD(msg=False))

        if status == -1:
            # If infeasible
            node.obj = math.inf
        elif status == 1:
            # If optimal
            node.obj = self.model.objective.value()
            node.solution = {
                var.name: var.value() for var in self.model.variables()
            }
        elif status == -2:
            # If unbounded
            node.obj = -math.inf
            self.bounds.update_best_obj(-math.inf)
        else:
            raise ValueError(f"Unexpected status code {status}")

        # If node is cutoff
        if node.obj > self.bounds.best_obj:
            self.print_inner_node_progress(node)
            return

        # If the solution is integer
        if status == 1:
            if self.is_solution_integer():
                self.bounds.update_best_obj(node.obj)
                self.z = node.obj
                self.solution = node.solution
            else:
                self.unexplored_list.insert(node)

                # Update best bound with the objective of the best unexplored
                best_lb_node = self.unexplored_list.peek()

                if not best_lb_node:
                    raise ValueError("Node must exist")

                self.bounds.update_best_bound(best_lb_node.obj)

        # Print progress to screen
        if node.parent:
            self.print_inner_node_progress(node)

    def get_objective(self) -> float:
        """Returns the best objective value found.

        Returns:
            float: The best objective value.
        """
        return self.z

    def get_solution(self) -> Optional[Dict[str, float]]:
        """Returns the optimal solution found.

        Returns:
            Optional[Dict[str, float]]: The optimal solution.
        """
        return self.solution

    def optimize(self) -> None:
        """Runs the branch-and-bound algorithm."""
        # Solve root node
        self.solve_node(self.tree.root)
        previous_node = self.tree.root

        self.print_root_node_progress()

        if not self.unexplored_list.is_empty:
            self.print_headers()

        # Intialise status
        status = self.check_convergence()

        # If list of unexplored is not empty, branch
        while not self.unexplored_list.is_empty:
            # Look for next node to solve (best bound search)
            selected_node = self.unexplored_list.pop()

            if selected_node.obj >= self.bounds.best_obj:
                raise ValueError("Objective value exceeds bounds")

            # Update model to the selected_node
            self.traverse(previous_node, selected_node)

            # Branch on selected node
            self.branch(selected_node)

            if not selected_node.left or not selected_node.right:
                raise ValueError("Node must have two children")

            # Solve left child node
            self.traverse(selected_node, selected_node.left)
            self.solve_node(selected_node.left)

            # Solve right child node
            self.traverse(selected_node.left, selected_node.right)
            self.solve_node(selected_node.right)

            # Check if we found the optimal solution or proved infeasibility
            status = self.check_convergence()
            if status != 0:
                break

            # Make current node the previous node
            previous_node = selected_node

        # Check optimization status
        if status == BB_UNBOUNDED:
            print("\nMIP Infeasible. No solution found.")
        elif status == BB_INFEASIBLE:
            print("\nMIP Unbounded. Best objective=inf")
        else:
            # If must have converged in the root
            if not self.check_convergence():
                raise ValueError("Unexpected best objective")

            # Print solution
            if status == BB_NODE_LIMIT:
                print("Node limit reached")
            elif status != BB_OPTIMAL:
                raise ValueError("Solution must be optimal at this point")

            self.print_mip_solution()
