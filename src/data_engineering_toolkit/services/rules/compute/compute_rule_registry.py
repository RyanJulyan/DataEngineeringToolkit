import heapq
import logging
from collections import defaultdict
from typing import Any, Dict, List

# Services
from data_engineering_toolkit.services.rules.compute.base_compute_rule import (
    BaseComputeRule,
)


class ComputeRuleRegistry:
    """
    A class for managing a collection of rules and their dependencies.

    This class acts as a registry for storing rules and managing their dependencies and execution order.
    It supports adding rules, checking for circular dependencies, and applying all rules to a given dataset.

    Attributes:
        rules (Dict[str, BaseComputeRule]): A mapping of rule names to rule instances.
        graph (defaultdict[List[str]]): A dependency graph representing rule dependencies.
        indegree (defaultdict[int]): A mapping of rule names to their indegree (number of dependencies).
        priority_queue (List[BaseComputeRule]): A priority queue of rules ordered by priority and addition order.
        addition_order (Dict[str, int]): A mapping of rule names to their addition order.
        rule_counter (int): A counter to keep track of the addition order of rules.
        raise_error_on_rule_apply_exception (bool): A flag to determine if an error should be raised when a rule application fails.

    Methods:
        add_rule(rule: BaseComputeRule): Adds a rule to the registry, resolving its dependencies.
        apply_all(data: Any) -> Any: Applies all rules in the registry to the given data.
    """

    def __init__(
        self,
        raise_error_on_rule_apply_exception: bool = False,
    ):
        """
        Initializes a new instance of the ComputeRuleRegistry class.

        Args:
            raise_error_on_rule_apply_exception (bool, optional): Flag to raise an exception if a rule application fails. Defaults to False.
        """

        self.rules: Dict[str, BaseComputeRule] = {}  # Rule name to rule mapping
        self.graph: defaultdict[List[str]] = defaultdict(list)  # Dependency graph
        self.indegree: defaultdict[int] = defaultdict(
            int
        )  # Number of dependencies for each rule
        self.priority_queue: List[
            BaseComputeRule
        ] = []  # Initialize an empty list for the priority queue
        self.addition_order: Dict[str, int] = {}  # Track the order of rule additions
        self.rule_counter: int = 0  # Counter to track addition order
        self.raise_error_on_rule_apply_exception: bool = (
            raise_error_on_rule_apply_exception
        )

    def _add_to_priority_queue(self, rule: BaseComputeRule) -> None:
        """Add a rule to the priority queue.

        Args:
            rule (BaseComputeRule): The rule to be added.
        """

        # Add rule to the priority queue with priority and addition order as key
        heapq.heappush(self.priority_queue, rule)

    def add_rule(self, rule: BaseComputeRule) -> None:
        """
        Add a rule to the registry, automatically resolving and adding its dependencies first.

        This method recursively adds each dependency of the provided rule to the registry
        before adding the rule itself. If a dependency is not already present in the registry,
        it is added by recursively calling this 'add_rule' method. After all dependencies are resolved,
        the rule is added or updated in the registry using '_add_or_update_rule'.

        Note: This method can lead to deep recursion if the dependency graph is large or complex.
        It assumes that the dependency graph does not contain cycles.

        Parameters:
        - rule (BaseComputeRule): The rule to be added to the registry. The rule must have a 'depends_on' attribute,
                       which is a list of other Rule instances that it depends on.

        Side Effects:
        - Resolves and adds dependencies of the rule to the registry.
        - Modifies the registry by adding or updating the rule.
        - Potentially modifies internal structures such as dependency graphs and priority queues through
          '_add_or_update_rule'.
        """

        # Resolve and add dependencies first
        for dependency in rule.depends_on:
            if dependency.name not in self.rules:
                self.add_rule(dependency)  # Recursively add dependencies

        # Now add the rule itself
        self._add_or_update_rule(rule)

    def _add_or_update_rule(self, rule: BaseComputeRule) -> None:
        """
        Add a new rule to the registry or update an existing rule based on its name.

        This method increments the rule counter and assigns it to the 'addition_order' of the rule.
        If the rule is new (not already in the registry), it is added. If it exists, its properties
        such as 'priority', 'depends_on', and 'args' are updated with the values from the provided rule.

        Additionally, the method updates the dependency graph and the indegree of each rule. If a rule
        does not have any dependencies (i.e., its indegree is zero), it is added to a priority queue.

        Parameters:
        - rule (BaseComputeRule): The rule to add or update. It must have attributes like 'name', 'priority',
                       'depends_on', and 'args'.

        Side Effects:
        - Increments 'self.rule_counter'.
        - Modifies 'self.rules', 'self.graph', and 'self.indegree'.
        - Potentially modifies a priority queue if the rule has no dependencies.
        """

        self.rule_counter += 1
        rule.addition_order = self.rule_counter

        if rule.name not in self.rules:
            self.rules[rule.name] = rule
        else:
            existing_rule = self.rules[rule.name]
            existing_rule.priority = rule.priority
            existing_rule.depends_on = rule.depends_on
            existing_rule.args = rule.args

        # Update the graph and indegree
        for dependency in rule.depends_on:
            self.graph[dependency.name].append(rule.name)
            self.indegree[rule.name] += 1

        if self.indegree[rule.name] == 0:
            self._add_to_priority_queue(rule)

    def _has_circular_dependency(self) -> bool:
        """Check for circular dependencies in the rule graph.

        Returns:
            bool: True if a circular dependency is found, False otherwise.
        """

        # Perform a topological sort of the graph

        def visit(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if visit(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited, rec_stack = set(), set()
        for rule_name in self.rules:
            if rule_name not in visited and visit(rule_name, visited, rec_stack):
                return True
        return False

    def check_for_circular_dependency(self) -> None:
        """
        Checks for circular dependencies within the rule graph.

        This method assesses the rule graph managed by the instance to identify any
        circular dependencies. A circular dependency occurs when a rule or a series of
        rules depend on each other, directly or indirectly, forming a loop in the dependency graph.

        Raises:
            Exception: If a circular dependency is detected in the rule graph.
        """
        if self._has_circular_dependency():
            raise Exception("Circular dependency detected")

    def apply_all(self, data: Any) -> Any:
        """Apply all rules in the registry to the given data.

        Args:
            data (Any): The data to which the rules are applied.

        Raises:
            Exception: If a rule fails to apply and raise_error_on_rule_apply_exception is set to True, or if a circular dependency is detected.

        Returns:
            Any: The result of applying all rules to the data.
        """

        self.check_for_circular_dependency()

        # Apply all rules in the priority queue
        while self.priority_queue:
            rule = heapq.heappop(self.priority_queue)  # Get the rule object directly

            try:
                rule.validate(data)  # Validate data
                data = rule.apply(data)  # Capture the updated data
            except Exception as e:
                logging.error(f"Error applying rule '{rule.name}': {e}")
                if self.raise_error_on_rule_apply_exception:
                    raise

            for dependent in self.graph[rule.name]:
                self.indegree[dependent] -= 1
                if self.indegree[dependent] == 0:
                    self._add_to_priority_queue(self.rules[dependent])

        return data
