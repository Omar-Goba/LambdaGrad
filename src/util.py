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


class LossFunction(str, Enum):
    BCE = "binary_cross_entropy"


### ~~~ STATE MANAGEMENT ~~~ ###
TARGET_COLUMN: str = "target"
RANDOM_SEED: int = 42
RATIOS: dict[str, float] = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15,
}
BATCH_SIZE: int = 100
EPOCHS: int = 10000
LEARNING_RATE: float = 0.1


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


def binary_cross_entropy(y_true: tensor_t, y_pred: tensor_t) -> float:
    """
    Compute the Binary Cross-Entropy loss.
    Args:
        y_true (tensor_t): True labels.
        y_pred (tensor_t): Predicted labels.
    Returns:
        float: Binary Cross-Entropy loss.
    """
    ### avoid log(0) ###
    epsilon: float = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    ### compute BCE ###
    bce: float = (
        -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    ).astype(float)

    return bce


def binary_cross_entropy_derivative(y_true: tensor_t, y_pred: tensor_t) -> tensor_t:
    """
    Compute the derivative of the Binary Cross-Entropy loss.
    Args:
        y_true (tensor_t): True labels.
        y_pred (tensor_t): Predicted labels.
    Returns:
        tensor_t: Derivative of Binary Cross-Entropy loss.
    """
    raise DeprecationWarning(
        "Use bce_with_sigmoid_last_layer_grad instead for numerical stability."
    )
    ### avoid division by zero ###
    epsilon: float = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    ### compute derivative ###
    bce_derivative: tensor_t = (
        -((y_true / y_pred) - ((1 - y_true) / (1 - y_pred))) / y_true.shape[0]
    )

    return bce_derivative


### ~~~ LOSS DERIVATIVES ~~~ ###
def bce_with_sigmoid_last_layer_grad(y_true: tensor_t, y_pred: tensor_t) -> tensor_t:
    """
    Gradient of binary cross-entropy wrt the *pre-activation* z
    of the last sigmoid layer.

    For sigmoid + BCE, we have:
        dL/dz = (y_pred - y_true) / m
    """
    m = y_true.shape[0]
    return (y_pred - y_true) / m


LOSS_FUNCTIONS: dict[LossFunction, Callable[[tensor_t, tensor_t], float]] = {
    LossFunction.BCE: binary_cross_entropy,
}

LOSS_DERIVATIVES: dict[LossFunction, Callable[[tensor_t, tensor_t], tensor_t]] = {
    LossFunction.BCE: bce_with_sigmoid_last_layer_grad,
}
