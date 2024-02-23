import inspect
import re
from dataclasses import dataclass
from types import FunctionType
import copy

import inflect
from statemachine import State, StateMachine

from data_engineering_toolkit.services.state_machine.state_machine_config_model import StateMachineConfig

p = inflect.engine()


@dataclass
class ManagedStateMachine:
  """A wrapper for a StateMachine instance, providing additional metadata and management features."""
  name: str
  config: StateMachineConfig
  state_machine: StateMachine = None

  def __post_init__(self):
    # Create States
    states = {}
    for state_def in self.config.states:
      states[state_def.name] = State(
          name=state_def.name,
          value=state_def.value,
          initial=state_def.is_initial,
          final=state_def.is_final,
          enter=state_def.enter,
          exit=state_def.exit,
      )

    # Create Methods:
    methods = {}
    for method in self.config.transitions:
      if method.before is not None:
        methods[method.before.__name__] = method.before
      if method.on is not None:
        methods[method.on.__name__] = method.on
      if method.after is not None:
        methods[method.after.__name__] = method.after
      if method.cond is not None:
        for cond in method.cond:
          methods[cond.__name__] = cond
      if method.unless is not None:
        for unless in method.unless:
          methods[unless.__name__] = unless
      if method.validators is not None:
        for validator in method.validators:
          methods[validator.__name__] = validator

    # Create Transitions
    transitions = {}
    for transition_def in self.config.transitions:
      conds = ""
      if transition_def.cond is not None:
        conds = ",".join(
            [f"'{cond.__name__}'" for cond in transition_def.cond])

      unless = ""
      if transition_def.unless is not None:
        unless = ",".join(
            [f"'{unless.__name__}'" for unless in transition_def.unless])

      validators = ""
      if transition_def.validators is not None:
        validators = ",".join([
            f"'{validator.__name__}'"
            for validator in transition_def.validators
        ])

      # Create transitions with direct callable references and conditional inclusion
      transition_args = {
          'before':
          transition_def.before,
          'on':
          transition_def.on,
          'after':
          transition_def.after,
          'cond':
          transition_def.cond if transition_def.cond else None,
          'unless':
          transition_def.unless if transition_def.unless else None,
          'validators':
          transition_def.validators if transition_def.validators else None,
      }

      # Filter out None values
      filtered_transition_args = {
          k: v
          for k, v in transition_args.items() if v is not None
      }

      transitions[transition_def.name] = states[transition_def.source].to(
          states[transition_def.destination], **filtered_transition_args)

    # Dynamically create a new class that inherits from StateMachine
    # and add states and methods to it
    new_class_name = self.to_class_name(self.name)

    self.state_machine = type(new_class_name, (StateMachine, ), {
        **states,
        **transitions,
        **methods,
    })

    # Now, self.state_machine has everything from the config

  def convert_numbers_to_words(self, s):
    # Function to replace each match with its word equivalent
    def replace_with_words(match):
      number = int(match.group())
      words = p.number_to_words(number, andword="")
      # Split into words, capitalize each, and join without spaces for CamelCase
      words = ''.join(word.capitalize()
                      for word in words.replace('-', ' ').split())
      return words

    # Replace all numeric sequences with their word equivalents
    return re.sub(r'\d+', replace_with_words, s)

  def to_class_name(self, s):
    # Convert all numbers to words
    s = self.convert_numbers_to_words(s)

    # CamelCase conversion: split by any non-alphanumeric, capitalize each, and join
    s = re.sub(r'[^a-zA-Z0-9]+', ' ',
               s)  # Replace non-alphanumeric with space for splitting
    words = s.split()
    class_name = ''.join(word[0].upper() + word[1:] for word in words if word)

    return class_name
