### ~~~ GLOBAL IMPORTS ~~~ ###
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
import seaborn as sns
import pandas as pd
import numpy as np

### ~~~ STATE MANAGEMENT ~~~ ###
np.random.seed(42)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
BATCH_SIZE = 100
EPOCHS = 50
LEARNING_RATE = 0.01


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


def feature_engineering(df: pd.DataFrame, do_plots: bool = True) -> pd.DataFrame:
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
    **3. Missing Value Handling (Req. 2.3)**
        - Detects missing values and prints counts.
        - If missing data exists, imputes numerical columns with their column means.
        - Prints explanation of the chosen imputation strategy.
    **4. Statistical Characterization (Req. 2.4)**
        - Computes extended descriptive statistics for all numerical features
          (count, mean, median, std, variance, min, max, range).
        - Prints the resulting statistics table.
    Args:
        df : pd.DataFrame
            The raw dataset, expected to include a column named "target".
        do_plots : bool, optional
            Whether to generate and display visualizations. Default is True.
    Returns:
        pd.DataFrame
            A cleaned and preprocessed copy of the dataset after handling missing
            values and performing statistical analysis. No new engineered features
            are added; this function focuses on profiling and preparing the data.
    """
    ############################################################
    ### Requirement 2.1: Features Identification and Summary ###
    ############################################################

    ### copy the df to avoid annoying warnings ###
    df_copy = df.copy()

    ### set up the seaborn ###
    sns.set(style="whitegrid")

    ### Enhanced Feature Summary Table ###
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
    if do_plots:
        print("Enhanced Feature Summary Table:")
        print(feature_summary)

    ################################################
    ### Requirement 2.2: Visual Data Exploration ###
    ################################################

    ### drop the target for visualization purposes ###
    df_features = df_copy.drop(columns="target")  # Exclude target for histograms

    #### 1. Histograms for all features ####
    df_features.hist(
        bins=15, figsize=(20, 15), layout=(6, 5), color="skyblue", edgecolor="black"
    )
    if do_plots:
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
            data=df, x=x_feat, y=y_feat, hue="target", palette={0: "red", 1: "green"}
        )
        plt.title(f"Scatter Plot: {x_feat} vs {y_feat}")
        plt.show()

    ### 3. Correlation Heatmap ###
    if do_plots:
        plt.figure(figsize=(15, 12))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap of Features", fontsize=18)
        plt.show()

    ### 4. Box Plots for All Numeric Features ###
    ## Selecting numeric columns (all features except target) ##
    numerical_cols = df_copy.drop(columns="target").columns
    n_plots = len(numerical_cols)  # total features
    n_cols = 5  # number of columns in grid
    n_rows = int(np.ceil(n_plots / n_cols))  # number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()  # flatten for easy iteration

    ## Looping through each numeric feature and create boxplot ##
    for idx, col in enumerate(numerical_cols):
        axes[idx].boxplot(
            df_copy[col].dropna(),
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
    if do_plots:
        plt.tight_layout()
        plt.show()

    ### ################################################
    ### **Requirement 2.3: Handling Missing Values** ###
    ### ################################################

    ### Check for missing values ###
    missing_values = df_copy.isnull().sum()
    if do_plots:
        print("Missing values per feature:")
        print(missing_values)

    ### Check if any column has missing values ###
    if missing_values.sum() == 0:
        if do_plots:
            print("\nNo missing values found. No imputation or removal needed.")
    else:
        ## if there were missing values, we would fill them with the mean of the column ##
        df_copy.fillna(df_copy.mean(), inplace=True)
        print("\nMissing values were imputed with column mean.")

    ### Justification ###
    """
    We decided to fill any missing numeric values with the mean of each feature (mean imputation). This approach is simple
    and practical. It keeps the overall distribution of the data intact and ensures we don't lose any
    rows, so all available information is used for training the neural network. Since this dataset has 
    no missing values, using the mean is an effective and reasonable choice.
    """

    #######################################################################
    ### Requirement 2.4: Statistical Characterization of Numerical Data ###
    #######################################################################

    ### Select numeric features (exclude target) ###
    numeric_features = df_copy.drop(columns="target")

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

    return df_copy


def data_partioning(df: pd.DataFrame) -> tuple:
    """"""
    # Get all columns except last one (features)
    x = df.iloc[:, :-1].values

    # Get only last column (target)
    y = df.iloc[:, -1].values

    # Set random seed for reproducibility
    # Ensures that the random operations produce the same result every time we run the code.

    # Total number of samples
    n_samples = x.shape[0]

    # Shuffle the indices
    indices = np.random.permutation(n_samples)

    # Compute split sizes
    train_size = int(TRAIN_RATIO * n_samples)
    val_size = int(VAL_RATIO * n_samples)

    # Split indices
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    # Create datasets
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    # Print shapes to confirm
    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Validation set: {x_val.shape}, {y_val.shape}")
    print(f"Testing set: {x_test.shape}, {y_test.shape}")

    return (x_train, y_train, x_val, y_val, x_test, y_test)


def mini_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> list:
    """
    Generate mini-batches from the dataset.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        batch_size (int): Size of each mini-batch.
    Returns:
        list: A list of tuples, each containing a mini-batch of (X_batch, y_batch).
    """
    n_samples = X.shape[0]
    mini_batches = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        mini_batches.append((X_batch, y_batch))

    return mini_batches


class NeuralNetwork:
    """
    A very simple fully-connected Neural Network built completely from scratch.

    Architecture:
        [N input neurons] → [10 hidden neurons] → [output layer]

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int, optional
        Number of neurons in the hidden layer. Default is 10.
    output_size : int, optional
        Number of output neurons. For binary classification this is typically 1.
    learning_rate : float, optional
        Learning rate used for gradient descent updates.
    activation : str, optional
        Activation function for the hidden layer. One of:
        ["sigmoid", "relu", "tanh"].
    task : str, optional
        Task type. Currently used for display/logging purposes
        (e.g., "classification").
    """

    ##########################################################
    ### Internal Activation Functions for the Neural Net  ###
    ##########################################################
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically-stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid function."""
        sig = NeuralNetwork.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (z > 0).astype(float)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """tanh activation."""
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of tanh."""
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Row-wise softmax for multi-class outputs."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 10,
        output_size: int = 1,
        learning_rate: float = 0.01,
        activation: str = "sigmoid",
        task: str = "classification",
    ) -> None:
        """
        Initialize the neural network with given architecture and hyperparameters.
        """

        #########################################
        ### Save Network Configuration       ###
        #########################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.task = task

        #########################################
        ### Choose Hidden Layer Activation    ###
        #########################################
        self.activation_name = activation

        if activation == "sigmoid":
            self.activation = NeuralNetwork.sigmoid
            self.activation_derivative = NeuralNetwork.sigmoid_derivative
        elif activation == "relu":
            self.activation = NeuralNetwork.relu
            self.activation_derivative = NeuralNetwork.relu_derivative
        elif activation == "tanh":
            self.activation = NeuralNetwork.tanh
            self.activation_derivative = NeuralNetwork.tanh_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")

        #########################################
        ### Weight Initialization             ###
        #########################################
        ## Input → Hidden ##
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(
            2.0 / input_size
        )
        self.bias_hidden = np.zeros((1, hidden_size))

        ## Hidden → Output ##
        self.weights_hidden_output = np.random.randn(
            hidden_size, output_size
        ) * np.sqrt(2.0 / hidden_size)
        self.bias_output = np.zeros((1, output_size))

        #########################################
        ### Training History Containers       ###
        #########################################
        self.train_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []

        #########################################
        ### Summary Printout                  ###
        #########################################
        print("✓ Neural Network initialized successfully")
        print(f"  Architecture: [{input_size} → {hidden_size} → {output_size}]")
        print(f"  Activation: {activation}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Task: {task}")


def main() -> int:
    """"""
    ### Section I ###
    df = load_data()

    ### Section II ###
    df = feature_engineering(df, do_plots=False)

    ### Section III ###
    ## Data Partitioning ##
    x_train, y_train, x_val, y_val, x_test, y_test = data_partioning(df)

    ## Batch Generation ##
    train_batches = mini_batches(x_train, y_train, BATCH_SIZE)

    ## Init the model ##
    # Determine number of classes #
    n_classes = len(np.unique(y_train))

    # Determine output size (binary vs multi-class) #
    output_neurons = n_classes if n_classes > 2 else 1

    # Create Neural Network instance #
    nn = NeuralNetwork(
        input_size=x_train.shape[1],
        hidden_size=10,
        output_size=output_neurons,
        learning_rate=LEARNING_RATE,
        activation="sigmoid",
        task="classification",
    )

    return 0


if __name__ == "__main__":
    exit(main())
