from dataclasses import dataclass
from typing import List

from data_engineering_toolkit.services.state_machine.state_definition_model import StateDefinition
from data_engineering_toolkit.services.state_machine.transition_definition_model import TransitionDefinition


@dataclass
class StateMachineConfig:
  """Configuration for a state machine, including its states and transitions."""
  states: List[StateDefinition]
  transitions: List[TransitionDefinition]

  def __post_init__(self):
    initial_states = [state for state in self.states if state.is_initial]

    if len(initial_states) > 1:
      raise ValueError("There must be exactly one or less initial state.")
