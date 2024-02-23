from dataclasses import dataclass, field
from typing import Dict

from statemachine import StateMachine

from data_engineering_toolkit.models.state_machine.state_machine_config_model import (
    StateMachineConfig,
)
from data_engineering_toolkit.services.state_machine.managed_state_machine import (
    ManagedStateMachine,
)


@dataclass
class StateMachineManager:
    """Manages multiple StateMachine instances, allowing for their creation, modification, and querying."""

    state_machines: Dict[str, StateMachine] = field(default_factory=dict)

    def create_state_machine(
        self, name: str, config: StateMachineConfig
    ) -> StateMachine:
        """Creates and registers a new StateMachine based on the provided configuration."""

        self.state_machines[name] = ManagedStateMachine(name, config).state_machine()

        return self.state_machines[name]

    def get_state_machine(self, name: str) -> StateMachine:
        """Fetch StateMachine by name."""

        return self.state_machines[name]

    def trigger_transition(self, name: str, transition_name: str, *args, **kwargs):
        return getattr(self.state_machines[name], transition_name)(*args, **kwargs)

    def get_current_state(self, name: str):
        """Returns the current state of the specified state machine."""
        return self.state_machines[name].current_state.id
