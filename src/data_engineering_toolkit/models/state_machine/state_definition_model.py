from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class StateDefinition:
    """Defines a state with a name and whether it's an initial/final state."""

    name: str
    value: Any = None
    is_initial: bool = False
    is_final: bool = False
    enter: Optional[Callable] = None
    exit: Optional[Callable] = None

    def __post_init__(self):
        if self.is_initial and self.is_final:
            raise ValueError("A state cannot be both initial and final.")
