from dataclasses import dataclass
from typing import Any

from data_engineering_toolkit.services.scheduling.job_builder import JobBuilder
from data_engineering_toolkit.services.scheduling.job_model import Job
from data_engineering_toolkit.services.scheduling.scheduler_model import Scheduler


@dataclass
class JobBuilderFactory:
  """Factory class for creating and configuring jobs using the JobBuilder.

  This factory class uses the provided job configuration (an instance of Job)
  to create and configure jobs via the JobBuilder class.

  Methods:
      create_job: Creates and configures a job based on the provided Job instance.
  """

  @staticmethod
  def create_job(scheduler: Scheduler, job: Job) -> Any:
    """
    Creates and configures a job using the JobBuilder.

    Uses the configuration details provided in the Job instance to set up
    the job's schedule, tags, time constraints, and the actual function to be executed.

    Args:
        scheduler (Scheduler): The scheduler to which the job will be added.
        job (Job): The configuration for the job, including scheduling details,
                   function to execute, and additional parameters.

    Returns:
        Any: The scheduled job as returned by the JobBuilder's 'do' method.
    """
    # Use the Job instance to configure the JobBuilder
    builder = JobBuilder(scheduler=scheduler).every(job.interval)

    getattr(builder, job.unit)()

    if job.latest is not None:
      builder.to(job.latest)

    if job.tags:
      builder.tag(*job.tags)

    if job.at_time_str:
      builder.at(job.at_time_str)

    if job.until_time:
      builder.until(job.until_time)

    return builder.do(job.job_func, *job.args, **job.kwargs)
