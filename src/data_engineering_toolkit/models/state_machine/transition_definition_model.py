from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class TransitionDefinition:
  """Defines a transition between states, including the source, destination, and the trigger method."""
  name: str
  source: str
  destination: str
  before: Optional[Callable] = None
  on: Optional[Callable] = None
  after: Optional[Callable] = None
  cond: Optional[List[Callable]] = None
  unless: Optional[List[Callable]] = None
  validators: Optional[List[Callable]] = None
