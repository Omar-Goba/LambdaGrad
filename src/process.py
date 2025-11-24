### ~~~ GLOBAL IMPORTS ~~~ ###
import pandas as pd

### ~~~ Local IMPORTS ~~~ ###
from util import tensor_t, RANDOM_SEED


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
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    ### compute the split indices ###
    split_indices: list = []
    cumulative_ratio: float = 0.0
    for ratio in ratios.values():
        cumulative_ratio += ratio
        split_index: int = int(cumulative_ratio * len(df))
        split_indices.append(split_index)

    ### split the dataframe ###
    splits: dict = {}
    previous_index: int = 0
    for i, (split_name, _) in enumerate(ratios.items()):
        split_index = split_indices[i]
        splits[split_name] = df.iloc[previous_index:split_index].to_numpy()
        previous_index = split_index

    ### assure all data is used ###
    assert previous_index == len(df), "Not all data was used in the splits."

    return splits
