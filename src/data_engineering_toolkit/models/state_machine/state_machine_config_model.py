from dataclasses import dataclass, field
from typing import List

from data_engineering_toolkit.models.state_machine.state_definition_model import (
    StateDefinition,
)
from data_engineering_toolkit.models.state_machine.transition_definition_model import (
    TransitionDefinition,
)
from data_engineering_toolkit.models.state_machine.cycleable_states_model import (
    CycleableStates,
)


@dataclass
class StateMachineConfig:
    """Configuration for a state machine, including its states and transitions."""

    states: List[StateDefinition]
    transitions: List[TransitionDefinition]
    cycleable_states: List[CycleableStates] = field(default_factory=list)

    def __post_init__(self):
        initial_states = [state for state in self.states if state.is_initial]

        if len(initial_states) > 1:
            raise ValueError("There must be exactly one or less initial state.")

        if self.cycleable_states is not None:
            self.validate_cycleable_states()

        # Validate that state names and transition names do not overlap
        self.validate_names_do_not_overlap()

    def validate_cycleable_states(self):
        state_names = {state.name for state in self.states}
        for chainable_state in self.cycleable_states:  # Iterate through dictionary
            invalid_transitions = [
                state
                for state in chainable_state.ordered_cycleable_states
                if state not in state_names
            ]
            if invalid_transitions:
                raise ValueError(
                    f"Invalid states in cycle '{chainable_state.name}': {invalid_transitions}. These do not match any state names provided in `states`: {[state.name for state in self.states]}."
                )

    def validate_names_do_not_overlap(self):
        state_names = {state.name for state in self.states}
        transition_names = {transition.name for transition in self.transitions}
        chainable_state_names = {
            chainable_state.name for chainable_state in self.cycleable_states
        }

        # Check for overlap between state names and transition names
        if not state_names.isdisjoint(transition_names):
            overlapping_names = state_names.intersection(transition_names)
            raise ValueError(
                f"State names and transition names must be unique. Overlapping names: {overlapping_names}"
            )

        # Check for overlap between chainable_state names and both state and transition names
        if not chainable_state_names.isdisjoint(state_names.union(transition_names)):
            overlapping_names = chainable_state_names.intersection(
                state_names.union(transition_names)
            )
            raise ValueError(
                f"Chainable state names must be unique and not overlap with state or transition names. Overlapping names: {overlapping_names}"
            )
