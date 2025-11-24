### ~~~ GLOBAL IMPORTS ~~~ ###
from dataclasses import dataclass
import numpy as np

### ~~~ Local IMPORTS ~~~ ###
from util import LossFunction, LOSS_FUNCTIONS, ActivationFunction
from layer import (
    Layer,
    create_layer,
    call_layer,
    layer_to_str,
)


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass
class Model:
    layers: list[Layer]
    loss_function: LossFunction
    learning_rate: float


def create_model(
    layers: list[Layer],
    loss_function: LossFunction,
    learning_rate: float = 0.001,
) -> Model:
    """
    Create a neural network model.
    Args:
        layers (list[Layer]): The layers of the model.
        loss_function (LossFunction): The loss function for the model.
        learning_rate (float): The learning rate for training.
    Returns:
        Model: The created neural network model.
    """
    ### insure that the dimensions match ###
    for i in range(1, len(layers)):
        if layers[i - 1].output_dim != layers[i].input_dim:
            raise ValueError(
                f"Layer {i} input dimension {layers[i].input_dim} does not match "
                f"previous layer output dimension {layers[i - 1].output_dim}"
            )

    ### create the model ###
    model: Model = Model(
        layers=layers,
        loss_function=loss_function,
        learning_rate=learning_rate,
    )

    return model


def call_model(model: Model, x: np.ndarray) -> np.ndarray:
    """
    Forward pass through the model.
    Args:
        model (Model): The neural network model.
        x (np.ndarray): The input tensor.
    Returns:
        np.ndarray: The output tensor after passing through the model.
    """
    ### make sure x has the right shape ###
    if x.shape[1] != model.layers[0].input_dim[0]:
        raise ValueError(
            f"Input tensor shape {x.shape} does not match model input dimension "
            f"{model.layers[0].input_dim}"
        )

    ### pass through each layer ###
    for layer in model.layers:
        x = call_layer(layer, x)

    return x


def compute_loss(model: Model, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the loss of the model.
    Args:
        model (Model): The neural network model.
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
    Returns:
        float: The computed loss.
    """
    loss_function = LOSS_FUNCTIONS[model.loss_function]
    loss: float = loss_function(y_true, y_pred)

    return loss


def model_to_str(model: Model) -> str:
    """
    Generate a string representation of the model.
    Args:
        model (Model): The neural network model.
    Returns:
        str: String representation of the model.
    """
    layer_strs: list[str] = [layer_to_str(layer) for layer in model.layers]
    model_str: str = (
        "Model(layers=[\n  " + ",\n  ".join(layer_strs) + f"\n], "
        f"loss_function={model.loss_function}, "
        f"learning_rate={model.learning_rate})"
    )

    return model_str


def main() -> int:
    model: Model = create_model(
        layers=[
            create_layer(
                input_dim=(3,),
                output_dim=(400,),
                activation=ActivationFunction.RELU,
                name="layer1",
            ),
            create_layer(
                input_dim=(400,),
                output_dim=(1,),
                activation=ActivationFunction.SIGMOID,
                name="layer2",
            ),
        ],
        loss_function=LossFunction.BCE,
        learning_rate=0.01,
    )
    x: np.ndarray = np.random.randn(4, 3)
    output: np.ndarray = call_model(model, x)
    print(f"loss: {compute_loss(model, np.array([[1], [0], [1], [0]]), output)}")
    # print("Model output:\n", output)
    # for layer in model.layers:
    #     print(layer.z)
    print(model_to_str(model))

    return 0


if __name__ == "__main__":
    exit(main())
