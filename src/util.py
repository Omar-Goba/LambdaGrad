### ~~~ GLOBAL IMPORTS ~~~ ###
from sklearn.datasets import load_breast_cancer
from enum import Enum
import pandas as pd
import numpy as np

### ~~~ TYPE DEFINITIONS ~~~ ###
from typing import Callable, TypeAlias

tensor_t: TypeAlias = np.ndarray
dim_t: TypeAlias = tuple[int, ...]


class ActivationFunction(str, Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LINEAR = "linear"


### ~~~ STATE MANAGEMENT ~~~ ###
RANDOM_SEED: int = 42
RATIOS: dict[str, float] = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15,
}


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


def relu(x: tensor_t) -> tensor_t:
    """
    Apply the ReLU activation function.
    Args:
        x (tensor_t): Input tensor.
    Returns:
        tensor_t: Output tensor after applying ReLU.
    """
    return np.maximum(0, x)


def relu_derivative(x: tensor_t) -> tensor_t:
    """
    Compute the derivative of the ReLU activation function.
    Args:
        x (tensor_t): Input tensor.
    Returns:
        tensor_t: Derivative of ReLU.
    """

    return (x > 0).astype(x.dtype)


def sigmoid(x: tensor_t) -> tensor_t:
    """
    Apply the Sigmoid activation function.
    Args:
        x (tensor_t): Input tensor.
    Returns:
        tensor_t: Output tensor after applying Sigmoid.
    """

    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: tensor_t) -> tensor_t:
    """
    Compute the derivative of the Sigmoid activation function.
    Args:
        x (tensor_t): Input tensor.
    Returns:
        tensor_t: Derivative of Sigmoid.
    """

    sig = sigmoid(x)
    return sig * (1 - sig)


def tanh(x: tensor_t) -> tensor_t:
    """
    Apply the Tanh activation function.
    Args:
        x (tensor_t): Input tensor.
    Returns:
        tensor_t: Output tensor after applying Tanh.
    """

    return np.tanh(x)


def tanh_derivative(x: tensor_t) -> tensor_t:
    """
    Compute the derivative of the Tanh activation function.
    Args:
        x (tensor_t): Input tensor.
    Returns:
        tensor_t: Derivative of Tanh.
    """

    return 1 - np.tanh(x) ** 2


def linear(x: tensor_t) -> tensor_t:
    """
    Apply the Linear activation function.
    Args:
        x (tensor_t): Input tensor.
    Returns:
        tensor_t: Output tensor after applying Linear.
    """

    return x


def linear_derivative(x: tensor_t) -> tensor_t:
    """
    Compute the derivative of the Linear activation function.
    Args:
        x (tensor_t): Input tensor.
    Returns:
        tensor_t: Derivative of Linear.
    """

    return np.ones_like(x)


ACTIVATION_FUNCTIONS: dict[ActivationFunction, Callable[[tensor_t], tensor_t]] = {
    ActivationFunction.RELU: relu,
    ActivationFunction.SIGMOID: sigmoid,
    ActivationFunction.TANH: tanh,
    ActivationFunction.LINEAR: linear,
}
ACTIVATION_DERIVATIVES: dict[ActivationFunction, Callable[[tensor_t], tensor_t]] = {
    ActivationFunction.RELU: relu_derivative,
    ActivationFunction.SIGMOID: sigmoid_derivative,
    ActivationFunction.TANH: tanh_derivative,
    ActivationFunction.LINEAR: linear_derivative,
}
