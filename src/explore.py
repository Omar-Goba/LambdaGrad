### ~~~ GLOBAL IMPORTS ~~~ ###
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

### ~~~ STATE DEFINITIONS ~~~ ###
sns.set(style="whitegrid")


def show_feature_summary(df: pd.DataFrame) -> None:
    """
    1. Feature Identification & Summary (Req. 2.1)
        - Generates a detailed summary table for all features, including:
          data types, total/unique values, missing values, sample values,
          min/max, mean, median, variance, and standard deviation.
        - Prints the summary to the console.
    Args:
        df : pd.DataFrame
            The raw dataset.
    Returns:
        None
    """
    feature_summary = pd.DataFrame(
        {
            "Feature": df.columns,
            "Data Type": [df[col].dtype for col in df.columns],
            "Total Values": [df[col].count() for col in df.columns],
            "Unique Values": [df[col].nunique() for col in df.columns],
            "Missing Values": [df[col].isnull().sum() for col in df.columns],
            "Sample Value": [df[col].iloc[0] for col in df.columns],
            "Min Value": [df[col].min() for col in df.columns],
            "Max Value": [df[col].max() for col in df.columns],
            "Mean": [df[col].mean() for col in df.columns],
            "Median": [df[col].median() for col in df.columns],
            "Std Dev": [df[col].std() for col in df.columns],
            "Variance": [df[col].var() for col in df.columns],
        }
    )

    ### some printing ###
    print("Enhanced Feature Summary Table:")
    print(feature_summary)

    return None


def show_visualizations(df: pd.DataFrame) -> None:
    """
    Generate and display visualizations for the dataset.
        - Histograms for all non-target features.
        - Scatter plots for selected feature pairs, colored by target class.
        - Correlation heatmap using Pearson correlations.
        - Box plots of all numerical features to reveal distribution shape and outliers.
    Args:
        df : pd.DataFrame
            The raw dataset.
    Returns:
        None
    """
    ### drop the target for visualization purposes ###
    df_features = df.drop(columns="target")

    #### 1. Histograms for all features ####
    df_features.hist(
        bins=15, figsize=(20, 15), layout=(6, 5), color="skyblue", edgecolor="black"
    )
    plt.suptitle("Feature Distributions", fontsize=20)
    plt.show()

    ### 2. Scatter Plots for Selected Feature Pairs ###
    feature_pairs = [
        ("mean radius", "mean texture"),
        ("mean area", "mean smoothness"),
        ("mean concavity", "mean compactness"),
    ]
    for x_feat, y_feat in feature_pairs:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df,
            x=x_feat,
            y=y_feat,
            hue="target",
            palette={0: "red", 1: "green"},
        )
        plt.title(f"Scatter Plot: {x_feat} vs {y_feat}")
        plt.show()

    ### 3. Correlation Heatmap ###
    plt.figure(figsize=(15, 12))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Features", fontsize=18)
    plt.show()

    ### 4. Box Plots for All Numeric Features ###
    ## Selecting numeric columns (all features except target) ##
    numerical_cols = df.drop(columns="target").columns
    n_plots = len(numerical_cols)  # total features
    n_cols = 5  # number of columns in grid
    n_rows = int(np.ceil(n_plots / n_cols))  # number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()  # flatten for easy iteration

    ## Looping through each numeric feature and create boxplot ##
    for idx, col in enumerate(numerical_cols):
        axes[idx].boxplot(
            df[col].dropna(),
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
        )
        axes[idx].set_title(f"Box Plot: {col}", fontsize=10, fontweight="bold")
        axes[idx].set_ylabel(col, fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis="y")

    ## Removing empty subplots if total plots < grid cells ##
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])

    ### Adjust layout and show ###
    plt.tight_layout()
    plt.show()


def show_statistical_characterization(df: pd.DataFrame) -> None:
    """
    3. Statistical Characterization (Req. 2.4)
        - Computes extended descriptive statistics for all numerical features
          (count, mean, median, std, variance, min, max, range).
        - Prints the resulting statistics table.
    """
    ### Select numeric features (exclude target) ###
    numeric_features = df.drop(columns="target")

    ### Compute descriptive statistics ###
    stats_summary = numeric_features.describe().T

    ### Add median ###
    stats_summary["median"] = numeric_features.median()

    ### Add variance ###
    stats_summary["variance"] = numeric_features.var()

    ### Add range (max - min) ###
    stats_summary["range"] = numeric_features.max() - numeric_features.min()

    ### Reorder columns for clarity ###
    stats_summary = stats_summary[
        ["count", "mean", "median", "std", "variance", "min", "max", "range"]
    ]

    print("\nStatistical Characterization of Numerical Data:")
    print(stats_summary)

    return None


def feature_engineering(df: pd.DataFrame) -> int:
    """
    Perform feature exploration, visualization, and preprocessing on the input dataset.
    This function fulfills the following tasks:
    **1. Feature Identification & Summary (Req. 2.1)**
        - Generates a detailed summary table for all features, including:
          data types, total/unique values, missing values, sample values,
          min/max, mean, median, variance, and standard deviation.
        - Prints the summary to the console.
    **2. Visual Data Exploration (Req. 2.2)**
        - Histograms for all non-target features.
        - Scatter plots for selected feature pairs, colored by target class.
        - Correlation heatmap using Pearson correlations.
        - Box plots of all numerical features to reveal distribution shape and outliers.
    **3. Statistical Characterization (Req. 2.4)**
        - Computes extended descriptive statistics for all numerical features
          (count, mean, median, std, variance, min, max, range).
        - Prints the resulting statistics table.
    Args:
        df : pd.DataFrame
            The raw dataset, expected to include a column named "target".
            Whether to generate and display visualizations. Default is True.
    Returns:
        int: Status code (0 for success).
    """
    ### set up the seaborn ###

    ### Enhanced Feature Summary Table ###
    show_feature_summary(df)

    ### Visual Data Exploration ###
    show_visualizations(df)

    ### Statistical Characterization ###
    show_statistical_characterization(df)

    return 0
