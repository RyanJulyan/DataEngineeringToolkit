import datetime
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Optional, Union

import schedule

from scheduler_model import Scheduler


@dataclass
class JobBuilder:
  """
  A builder class for creating and configuring a job in a fluent style.

  This class facilitates the creation of a job by providing methods to specify
  various scheduling options like frequency, time, and additional settings. 

  Attributes:
      scheduler (Scheduler): The scheduler to which the job will be added.
      job (Optional[schedule.Job]): An instance of a schedule.Job, 
                                    which is built and configured through this class.

  Methods:
      every: Sets the interval for the job.
      to: Sets the latest time to run the job.
      seconds: Sets the job to run every 'n' seconds.
      minutes: Sets the job to run every 'n' minutes.
      hours: Sets the job to run every 'n' hours.
      days: Sets the job to run every 'n' days.
      weeks: Sets the job to run every 'n' weeks.
      [day of the week]: Sets the job to run on a specific day of the week.
      tag: Adds tags to the job for identification.
      at: Sets a specific time of day for the job to run.
      until: Sets an ending point for the job.
      do: Finalizes the job with the function to execute and its arguments.
  """
  scheduler: Scheduler
  job: Optional[schedule.Job] = None

  def every(self, interval: int = 1) -> 'JobBuilder':
    """ Set the frequency of the job in terms of an interval. """
    self.job = self.scheduler.scheduler.every(interval)
    return self

  def to(self, latest: int) -> 'JobBuilder':
    """ Set the latest time to run the job. """
    self.job.to(latest)
    return self

  def seconds(self) -> 'JobBuilder':
    """ Set the job to run every 'n' seconds. """
    self.job.seconds
    return self

  def minutes(self) -> 'JobBuilder':
    """ Set the job to run every 'n' minutes. """
    self.job.minutes
    return self

  def hours(self) -> 'JobBuilder':
    """ Set the job to run every 'n' hours. """
    self.job.hours
    return self

  def days(self) -> 'JobBuilder':
    """ Set the job to run every 'n' days. """
    self.job.days
    return self

  def weeks(self) -> 'JobBuilder':
    """ Set the job to run every 'n' weeks. """
    self.job.weeks
    return self

  def monday(self) -> 'JobBuilder':
    """ Schedule the job to run every Monday. """
    self.job.monday
    return self

  def tuesday(self) -> 'JobBuilder':
    """ Schedule the job to run every Tuesday. """
    self.job.tuesday
    return self

  def wednesday(self) -> 'JobBuilder':
    """ Schedule the job to run every Wednesday. """
    self.job.wednesday
    return self

  def thursday(self) -> 'JobBuilder':
    """ Schedule the job to run every Thursday. """
    self.job.thursday
    return self

  def friday(self) -> 'JobBuilder':
    """ Schedule the job to run every Friday. """
    self.job.friday
    return self

  def saturday(self) -> 'JobBuilder':
    """ Schedule the job to run every Saturday. """
    self.job.saturday
    return self

  def sunday(self) -> 'JobBuilder':
    """ Schedule the job to run every Sunday. """
    self.job.sunday
    return self

  def tag(self, *tags: Hashable) -> 'JobBuilder':
    """ Add tags to the job for easier identification and management. """
    self.job.tag(tags)
    return self

  def at(self, time_str: str) -> 'JobBuilder':
    """ Set a specific time of day for the job to run. """
    self.job.at(time_str)
    return self

  def until(
      self,
      until_time: Union[datetime.datetime, datetime.timedelta, datetime.time,
                        str],
  ) -> 'JobBuilder':
    """ Set an ending point for the job, either as a date/time or a duration. """
    self.job.until(until_time)
    return self

  def do(self, job_func, *args, **kwargs) -> schedule.Job:
    """ Finalize the job with the function to execute and its arguments. """
    self.job.do(job_func, *args, **kwargs)
    return self.job
