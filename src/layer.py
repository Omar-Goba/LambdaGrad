### ~~~ GLOBAL IMPORTS ~~~ ###
from dataclasses import dataclass
import numpy as np

### ~~~ Local IMPORTS ~~~ ###
from util import dim_t, tensor_t, ActivationFunction, ACTIVATION_FUNCTIONS


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass
class Layer:
    name: str
    input_dim: dim_t
    output_dim: dim_t
    activation: ActivationFunction
    weights: tensor_t
    bias: tensor_t
    x: tensor_t | None
    z: tensor_t | None


def create_layer(
    input_dim: dim_t,
    output_dim: dim_t,
    activation: ActivationFunction,
    name: str = "layer",
) -> Layer:
    """
    Create a neural network layer with initialized weights and bias.
    the weights are initialized using He initialization
    Args:
        name (str): The name of the layer.
        input_dim (dim_t): The input dimensions of the layer.
        output_dim (dim_t): The output dimensions of the layer.
        activation (ActivationFunction): The activation function for the layer.
    Returns:
        Layer: The created neural network layer.
    """
    ### initialize weights and bias ###
    weights: tensor_t = np.random.randn(input_dim[0], output_dim[0]) * np.sqrt(
        2.0 / input_dim[0]
    )
    bias: tensor_t = np.zeros((1, output_dim[0]))

    ### create the layer ###
    layer: Layer = Layer(
        name=name,
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
        weights=weights,
        bias=bias,
        x=None,
        z=None,
    )

    return layer


def call_layer(layer: Layer, x: tensor_t) -> tensor_t:
    """
    Perform a forward pass through the layer.
    Args:
        layer (Layer): The neural network layer.
        x (tensor_t): The input tensor to the layer.
    Returns:
        tensor_t: The output tensor after applying the layer's weights, bias, and activation function.
    """
    ### compute the linear combination ###
    z: tensor_t = np.dot(x, layer.weights) + layer.bias

    ### apply the activation function ###
    a: tensor_t = ACTIVATION_FUNCTIONS[layer.activation](z)

    ### store intermediate values ###
    layer.x = x
    layer.z = z

    return a


def layer_to_str(layer: Layer) -> str:
    """
    Generate a string representation of the layer.
    Args:
        layer (Layer): The neural network layer.
    Returns:
        str: String representation of the layer.
    """
    return (
        f"Layer(name={layer.name}, input_dim={layer.input_dim}, "
        f"output_dim={layer.output_dim}, activation={layer.activation})"
    )


def main() -> int:
    """"""
    batch_size, features, outputs = 4, 8, 10
    x: tensor_t = np.random.randn(batch_size, features)
    layer = create_layer(
        input_dim=(features,),
        output_dim=(outputs,),
        activation=ActivationFunction.RELU,
        name="hidden_layer_1",
    )
    output: tensor_t = call_layer(layer, x)
    print(layer_to_str(layer))
    return 0


if __name__ == "__main__":
    exit(main())
