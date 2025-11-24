### ~~~ GLOBAL IMPORTS ~~~ ###
import pandas as pd
import numpy as np

### ~~~ Local IMPORTS ~~~ ###
from util import tensor_t, RNG, TARGET_COLUMN


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """"""
    ### copy the df to avoid annoying warnings ###
    df_copy = df.copy()

    ### get all columns with missing values ###
    missing_cols: list[str] = df_copy.columns[df_copy.isnull().any()].tolist()

    ### get the dtypes of those columns ###
    missing_dtypes: dict[str, str] = {
        col: str(df_copy[col].dtype) for col in missing_cols
    }

    ### Impute missing values with column means/modes ###
    for col, dtype in missing_dtypes.items():
        if "float" in dtype or "int" in dtype:
            mean_value = df_copy[col].mean()
            df_copy[col].fillna(mean_value, inplace=True)
        else:
            mode_value = df_copy[col].mode()[0]
            df_copy[col].fillna(mode_value, inplace=True)

    return df_copy


def split(df: pd.DataFrame, ratios: dict[str, float]) -> dict[str, tensor_t]:
    """"""
    ### insure the ratios sum to 1.0 ###
    total_ratio: float = sum(ratios.values())
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    ### shuffle the dataframe ###
    df = df.sample(frac=1, random_state=RNG).reset_index(drop=True)

    ### compute the split indices ###
    split_indices: list = []
    cumulative_ratio: float = 0.0
    split_index: int
    for ratio in ratios.values():
        cumulative_ratio += ratio
        split_index = int(cumulative_ratio * len(df))
        split_indices.append(split_index)

    ### split the dataframe ###
    splits: dict = {}
    previous_index: int = 0
    for i, (split_name, _) in enumerate(ratios.items()):
        split_index = split_indices[i]
        split_df: pd.DataFrame = df.iloc[previous_index:split_index]

        ### separate features and target ###
        X: tensor_t = split_df.drop(columns=[TARGET_COLUMN]).to_numpy()
        y: tensor_t = split_df[TARGET_COLUMN].to_numpy().reshape(-1, 1)

        ### store the split ###
        splits[split_name] = (X, y)

        ### update previous index ###
        previous_index = split_index

    ### assure all data is used ###
    assert previous_index == len(df), "Not all data was used in the splits."

    return splits


def batch_data(X: tensor_t, y: tensor_t, batch_size: int) -> tuple[tensor_t, tensor_t]:
    """
    Generate mini-batches from the dataset.
    Args:
        X (tensor_t): Feature matrix.
        y (tensor_t): Target vector.
        batch_size (int): Size of each mini-batch.
    Returns:
        tuple[nd.ndarray, ndarray]: a tuple containing mini-batches of features and targets.
    """
    ### init some stuff ###
    n_samples = X.shape[0]
    n_full_batches = n_samples // batch_size

    ### quick error check ###
    if n_full_batches == 0:
        raise ValueError("batch_size is larger than the number of samples.")

    ### trim off the remainder so all batches are equal ###
    x_trimmed = X[: n_full_batches * batch_size]
    y_trimmed = y[: n_full_batches * batch_size]

    ### split into equal-sized batches (no remainder) ###
    x_batches = np.array(np.split(x_trimmed, n_full_batches))
    y_batches = np.array(np.split(y_trimmed, n_full_batches))

    return (x_batches, y_batches)
