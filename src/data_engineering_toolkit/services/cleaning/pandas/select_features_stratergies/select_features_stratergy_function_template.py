from typing import Callable, Set

import pandas as pd


# target stratergy function signature
def select_features_stratergy_function_template(
    data: pd.DataFrame,
    target_column: str,
    n_features: int,
) -> Set[str]:
  """Function template for feature selection strategies.

  This function is a placeholder for the signature that all strategy functions should follow. 
  It takes a DataFrame, a target column name, and the number of features to select, and it 
  returns a set of selected feature names.

  Args:
      data (pd.DataFrame): The input data as a pandas DataFrame.
      target_column (str): The name of the target column in the data.
      n_features (int): The number of features to select.

  Returns:
      Set[str]: A set of selected feature names.
  """
  pass  # This function is just a placeholder for the signature, so no logic required


# Define a custom type for the function signature
StrategyFunction = Callable[[pd.DataFrame, str, int], Set[str]]
