from typing import Optional, Set

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor

from data_engineering_toolkit.services.cleaning.pandas.select_features_stratergies.select_features_stratergy_function_template import (
    select_features_stratergy_function_template, )
from data_engineering_toolkit.services.util.enforce_function_signature import (
    enforce_function_signature, )
from data_engineering_toolkit.services.util.get_default_class_parameters import get_default_class_parameters


@enforce_function_signature(
    target_signature=select_features_stratergy_function_template)
def k_neighbors_regressor_strategy(
    data: pd.DataFrame,
    target_column: str,
    n_features: int,
    n_neighbors: Optional[int] = None,
    scoring: Optional[str] = 'neg_mean_squared_error',
    model: Optional[KNeighborsRegressor] = None) -> Set[str]:
  """Selects features using the K Neighbors Regressor strategy.

  Implements a feature selection strategy based on the importance of features as
  determined by the K Neighbors Regressor model. If a model is not provided, a new one
  is created using the specified `n_neighbors` or default parameters. The function fits
  the model to the data, computes permutation importances, and selects the top `n_features`.

  Args:
      data (pd.DataFrame): The input dataset as a pandas DataFrame.
      target_column (str): The name of the target column in the data.
      n_features (int): The number of top features to select.
      n_neighbors (Optional[int]): The number of neighbors to use for the K Neighbors Regressor.
                                    If not provided, default parameters of the model are used.
      scoring (Optional[str]): Scoring method used for permutation importance. 
                               Defaults to 'neg_mean_squared_error'.
      model (Optional[KNeighborsRegressor]): An existing K Neighbors Regressor model. 
                                             If not provided, a new model is created.

  Returns:
      Set[str]: A set of names of the selected features.
  """
  if model is None and n_neighbors is None:
    n_neighbors = get_default_class_parameters(
        KNeighborsRegressor)['n_neighbors']
  if model is None:
    model = KNeighborsRegressor(n_neighbors=n_neighbors)

  model.fit(data, data[target_column])
  results = permutation_importance(model,
                                   data,
                                   data[target_column],
                                   scoring=scoring)
  df_importances = pd.DataFrame(data={
      'Attribute': data.columns,
      'Importance': results.importances_mean
  })
  df_importances = df_importances.sort_values(by='Importance', ascending=False)
  df_coefficients_importances = df_importances.head(n_features)
  return set(df_coefficients_importances['Attribute'].to_list())


if __name__ == "__main__":
  data = {
      "calories": [420, 380, 390, 420, 380],
      "duration": [50, 40, 45, 50, 40]
  }

  #load data into a DataFrame object:
  df = pd.DataFrame(data)

  print(
      k_neighbors_regressor_strategy(data=df,
                                     target_column="calories",
                                     n_features=2))
