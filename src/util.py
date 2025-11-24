### ~~~ GLOBAL IMPORTS ~~~ ###
from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_data() -> pd.DataFrame:
    """
    Dataset Selection: Breast Cancer Wisconsin (Diagnostic) (Classification)
    Args:
        None
    Returns:
        pd.DataFrame: The breast cancer dataset with features and target.
    """
    ### load up the data ###
    data: pd.DataFrame = load_breast_cancer()  # type: ignore

    ### put it into a df ###
    df = pd.DataFrame(data.data, columns=data.feature_names)

    ### add the target ###
    df["target"] = data.target

    return df
