from dataclasses import dataclass, field
from typing import Dict

import schedule

from job_model import Job


@dataclass
class Scheduler:
  """
  A class representing a scheduler that manages and executes jobs.

  This class encapsulates a `schedule.Scheduler` instance and provides
  a mapping of job names to `Job` objects, allowing for organized management
  and execution of scheduled tasks.

  Attributes:
      name (str): The name of the scheduler, used for identification.
      scheduler (schedule.Scheduler): An instance of `schedule.Scheduler` 
          which handles the actual scheduling and execution of tasks.
      jobs (Dict[str, Job]): A dictionary mapping job names to `Job` objects. 
          This allows for easy access and management of individual jobs.
  """
  name: str
  scheduler: schedule.Scheduler = field(
      default_factory=lambda: schedule.Scheduler())
  jobs: Dict[str, Job] = field(default_factory=dict)
