### ~~~ GLOBAL IMPORTS ~~~ ###
from functools import reduce
import pandas as pd
import numpy as np

### ~~~ Local IMPORTS ~~~ ###
from util import tensor_t, RNG, TARGET_COLUMN, FeatureSelectionMethod


def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types for each column in the DataFrame.
    - all numbers should be float32
    - categorical variables should be one hot encoded
    - target should always be densely encoded
    """
    ### copy the df to avoid annoying warnings ###
    df_copy = df.copy()

    ### convert numeric columns to float32 ###
    numeric_cols: list[str] = df_copy.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    for col in numeric_cols:
        df_copy[col] = df_copy[col].astype(np.float32)

    ### one-hot encode categorical variables (excluding target) ###
    categorical_cols: list[str] = df_copy.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != TARGET_COLUMN]
    df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)

    ### ensure target is densely encoded ###
    if df_copy[TARGET_COLUMN].dtype == "object" or str(
        df_copy[TARGET_COLUMN].dtype
    ).startswith("category"):
        df_copy[TARGET_COLUMN] = df_copy[TARGET_COLUMN].astype("category").cat.codes

    return df_copy


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


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """"""
    ### copy the df to avoid annoying warnings ###
    df_copy = df.copy()

    ### get all numeric columns ###
    numeric_cols: list[str] = df_copy.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    ### normalize each numeric column ###
    for col in numeric_cols:
        col_mean = df_copy[col].mean()
        col_std = df_copy[col].std()
        if col_std != 0:
            df_copy[col] = (df_copy[col] - col_mean) / col_std
        else:
            df_copy[col] = 0.0  # If std is 0, set all values to 0

    return df_copy


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """"""
    ### copy the df to avoid annoying warnings ###
    df_copy = df.copy()

    ### get all numeric columns ###
    numeric_cols: list[str] = df_copy.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    ### remove outliers for each numeric column ###
    for col in numeric_cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]

    ### reset index after removing rows ###
    df_copy.reset_index(drop=True, inplace=True)

    return df_copy


def feature_selection(
    df: pd.DataFrame,
    method: FeatureSelectionMethod = FeatureSelectionMethod.MUTUAL_INFORMATION,
) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    ### copy the df to avoid annoying warnings ###
    df_copy = df.copy()

    def mutual_information_discrete(x: pd.Series, y: pd.Series) -> float:
        """
        Compute Mutual Information I(X; Y) between two discrete Series.
        """
        ### joint distribution P(x, y)
        joint = pd.crosstab(x, y).astype(float)
        joint /= joint.values.sum()

        ### marginal distributions P(x), P(y)
        p_x = joint.sum(axis=1).values.reshape(-1, 1)
        p_y = joint.sum(axis=0).values.reshape(1, -1)

        ### compute MI with safe division + masking
        denom = p_x * p_y

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = joint.values / denom
            mask = (joint.values > 0) & np.isfinite(ratio)
            mi = (joint.values[mask] * np.log2(ratio[mask])).sum()

        return float(mi)

    if method == FeatureSelectionMethod.MUTUAL_INFORMATION:
        """
        Feature selection using Mutual Information. This is useful for both
        classification and regression tasks, as it captures any kind of dependency
        between features and the target variable.
        """
        ### compute MI for each feature against the target ###
        mi_scores_raw = {}

        for col in df_copy.columns:
            if col == TARGET_COLUMN:
                continue

            mi_scores_raw[col] = mutual_information_discrete(
                df_copy[col], df_copy[TARGET_COLUMN]
            )

        ### convert to Series for sorting ###
        mi_scores = pd.Series(mi_scores_raw)

        ### sort descending (higher MI = stronger relationship) ###
        mi_scores = mi_scores.sort_values(ascending=False)

        ### keep top 80% features ###
        num_keep = int(0.8 * len(mi_scores))
        threshold = mi_scores.iloc[num_keep - 1]

        selected_features = mi_scores[mi_scores >= threshold].index.tolist()

        ### final filtered DF ###
        df_filtered = df_copy[selected_features + [TARGET_COLUMN]]
        breakpoint()

        return df_filtered

    if method == FeatureSelectionMethod.CORRELATION:
        """
        Feature selection using Pearson Correlation Coefficient. This method is
        primarily useful for regression tasks where linear relationships are of interest.
        """
        ### compute correlation for each feature against the target ###
        corr_scores = (
            df_copy.drop(columns=[TARGET_COLUMN])  # exclude target from features
            .corrwith(df_copy[TARGET_COLUMN])  # Pearson corr with target
            .abs()  # use absolute correlation
            .sort_values(ascending=False)  # sort descending
        )

        ### keep top 80% features ###
        num_keep = int(0.8 * len(corr_scores))
        threshold = corr_scores.iloc[num_keep - 1]

        selected_features = corr_scores[corr_scores >= threshold].index.tolist()

        ### final filtered DF ###
        df_filtered = df_copy[selected_features + [TARGET_COLUMN]]
        breakpoint()

        return df_filtered


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is a pipeline that applies a series of data processing functions to the input DataFrame.
    1. Impute missing values
    2. Normalize numeric features
    3. Remove outliers
    Args:
        df (pd.DataFrame): Input DataFrame to be processed.
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    ### define the data processing functions ###
    data_processing_functions = [
        clean_dtypes,
        impute_data,
        normalize_data,
        remove_outliers,
        lambda x: feature_selection(x, method=FeatureSelectionMethod.CORRELATION),
    ]

    return reduce(lambda d, f: f(d), data_processing_functions, df)


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
