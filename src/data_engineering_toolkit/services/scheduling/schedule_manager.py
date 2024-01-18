from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import schedule

from job_builder_factory import JobBuilderFactory
from job_model import Job
from scheduler_model import Scheduler


@dataclass
class ScheduleManager:
  """
  A manager class that handles multiple schedulers and their associated jobs.

  This class provides functionality to create schedulers, add jobs to them,
  and manage these jobs (like retrieving, canceling, or running them).

  Attributes:
      schedulers (Dict[str, Scheduler]): A dictionary mapping scheduler names to 
          their respective `Scheduler` instances.
  """

  schedulers: Dict[str, Scheduler] = field(default_factory=dict)

  def create_scheduler(self,
                       scheduler_name: str,
                       jobs: Optional[Dict[str, Job]] = None):
    """
    Creates a new scheduler with the given name and optionally adds jobs to it.

    Args:
        scheduler_name (str): The name of the scheduler to create.
        jobs (Optional[Dict[str, Job]]): A dictionary of jobs to add to the scheduler.
    """
    if jobs is None:
      jobs = {}

    if scheduler_name not in self.schedulers:
      self.schedulers[scheduler_name] = Scheduler(
          name=scheduler_name, scheduler=schedule.Scheduler(), jobs=jobs)

    for job in jobs:
      JobBuilderFactory.create_job(
          scheduler=self.schedulers[scheduler_name],
          job=jobs[job],
      )

  def add_job(self, scheduler_name: str, job: Job):
    """
    Adds a new job to a named scheduler.

    If the scheduler does not exist, it will be created.

    Args:
        scheduler_name (str): The name of the scheduler to which the job will be added.
        job (Job): The job to be added.
    """
    if scheduler_name not in self.schedulers:
      self.create_scheduler(scheduler_name=scheduler_name)

    JobBuilderFactory.create_job(
        scheduler=self.schedulers[scheduler_name],
        job=job,
    )
    self.schedulers[scheduler_name].jobs[job.job_name] = job

  def get_all_jobs(self, scheduler_name: str):
    """ Retrieves all jobs from a specified scheduler. """
    self.schedulers[scheduler_name].scheduler.get_jobs()

  def cancel_all_jobs(self, scheduler_name: str):
    """ Cancels all jobs in a specified scheduler. """
    self.schedulers[scheduler_name].scheduler.clear()

  def get_jobs_by_tag(self, scheduler_name: str, tag: Hashable):
    """ Retrieves jobs by a specific tag from a specified scheduler. """
    self.schedulers[scheduler_name].scheduler.get_jobs(tag)

  def cancel_jobs_by_tag(self, scheduler_name: str, tag: Hashable):
    """ Cancels jobs by a specific tag in a specified scheduler. """
    self.schedulers[scheduler_name].scheduler.clear(tag)

  def time_until_the_next_execution(self,
                                    scheduler_name: str) -> Optional[float]:
    """
    Returns the time in seconds until the next job execution in a specified scheduler.

    Args:
        scheduler_name (str): The name of the
    Returns:
        float: The number of seconds until the next job execution.
    """
    return self.schedulers[scheduler_name].scheduler.idle_seconds

  def run_all(self, scheduler_name: str, delay_seconds: int = 0):
    """
    Immediately runs all jobs in a specified scheduler, with an optional delay.

    Args:
        scheduler_name (str): The name of the scheduler in which to run all jobs.
        delay_seconds (int): Optional delay in seconds before running the jobs.
    """
    self.schedulers[scheduler_name].scheduler.run_all(
        delay_seconds=delay_seconds)

  def run_pending(self, scheduler_name: str):
    """
    Immediately runs all jobs in a specified scheduler, that still need to be run.

    Args:
        scheduler_name (str): The name of the scheduler in which to run all jobs.
    """
    self.schedulers[scheduler_name].scheduler.run_pending(
        delay_seconds=delay_seconds)
