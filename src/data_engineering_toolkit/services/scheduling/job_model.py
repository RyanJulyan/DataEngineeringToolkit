import datetime
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional, Set, Union


@dataclass
class Job:
  """
  A data class representing a scheduled job with various configuration options.

  Attributes:
      job_name (str): Unique name of the job.
      job_func (Callable[[Any, Any], Any]): The function to be executed by the job.
      unit (Literal[...]): The unit of time for the job's frequency (e.g., seconds, minutes, days, etc.).
      interval (int): The interval at which the job should be run, in terms of the unit.
      latest (Optional[int]): The latest time within the interval that the job can run.
      tags (Set[Hashable]): A set of tags for categorizing or identifying the job.
      at_time_str (Optional[str]): Specific time of day (as a string) when the job should run.
      until_time (Optional[Union[datetime.datetime, datetime.timedelta, datetime.time, str]]): 
          Specifies when the job should end, either as a specific datetime, a duration, or a time.
      args (Iterable[Any]): Positional arguments to be passed to the job function.
      kwargs (Dict[str, Any]): Keyword arguments to be passed to the job function.

  Methods:
      __post_init__: A post-initialization method to automatically add the job_name to the tags set.
  """
  job_name: str
  job_func: Callable[[Any, Any], Any]
  unit: Literal["seconds", "minutes", "hours", "days", "weeks", "monday",
                "tuesday", "wednesday", "thursday", "friday", "saturday",
                "sunday", ]
  interval: int = 1
  latest: Optional[int] = None
  tags: Set[Hashable] = field(default_factory=set)
  at_time_str: Optional[str] = None
  until_time: Optional[Union[datetime.datetime, datetime.timedelta,
                             datetime.time, str]] = None
  args: Iterable[Any] = field(default_factory=list)
  kwargs: Dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    """
    Post-initialization method for the Job data class.

    Ensures that the `job_name` is automatically included in the `tags` set 
    for easy identification and categorization of the job.
    """
    self.tags.add(self.job_name)
