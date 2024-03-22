from dataclasses import dataclass, field
from typing import List


@dataclass
class CycleableStates:
    """Defines an attibute to cycle through specific states in an order."""

    name: str
    ordered_cycleable_states: List[str] = field(default_factory=list)
