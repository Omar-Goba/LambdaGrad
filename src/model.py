### ~~~ GLOBAL IMPORTS ~~~ ###
from dataclasses import dataclass
from tqdm import trange
import numpy as np

### ~~~ Local IMPORTS ~~~ ###
from util import (
    LossFunction,
    LOSS_FUNCTIONS,
    LOSS_DERIVATIVES,
    ActivationFunction,
    ACTIVATION_DERIVATIVES,
    tensor_t,
)
from layer import (
    Layer,
    Gradients,
    create_layer,
    call_layer,
    apple_gradients_to_layer,
    layer_to_str,
)
from callbacks import Callback, State


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


def call_model(model: Model, x: tensor_t) -> tensor_t:
    """
    Forward pass through the model.
    Args:
        model (Model): The neural network model.
        x (tensor_t): The input tensor.
    Returns:
        tensor_t: The output tensor after passing through the model.
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


def compute_loss(model: Model, y_true: tensor_t, y_pred: tensor_t) -> float:
    """
    Compute the loss of the model.
    Args:
        model (Model): The neural network model.
        y_true (tensor_t): The true labels.
        y_pred (tensor_t): The predicted labels.
    Returns:
        float: The computed loss.
    """
    loss_function = LOSS_FUNCTIONS[model.loss_function]
    loss: float = loss_function(y_true, y_pred)

    return loss


def apply_gradients(model: Model, grads_list: list[Gradients]) -> None:
    """
    Apply gradients to the model's layers.
    Args:
        model (Model): The neural network model.
        grads_list (list[Gradients]): List of gradients for each layer.
    """
    if len(grads_list) != len(model.layers):
        raise ValueError(
            f"Number of gradients {len(grads_list)} does not match number of "
            f"layers {len(model.layers)}"
        )

    for layer, grads in zip(model.layers, grads_list):
        apple_gradients_to_layer(layer, grads, model.learning_rate)


def compute_gradients(
    model: Model, x_batch: tensor_t, y_batch: tensor_t
) -> list[Gradients]:
    """"""
    ### forward pass to fill the cache ###
    y_pred: tensor_t = call_model(model, x_batch)

    ### compute loss derivative ###
    dL_dy: tensor_t = LOSS_DERIVATIVES[model.loss_function](y_batch, y_pred)

    ### prepare for the backward pass ###
    grads_list: list[Gradients] = []
    grad_next: tensor_t = dL_dy
    rev_layers: list[Layer] = model.layers[::-1]

    ### backward pass ###
    for idx, layer in enumerate(rev_layers):
        """
        Backward pass through a single layer.
        """
        ## get the cached values ##
        x: tensor_t | None = layer.x
        z: tensor_t | None = layer.z

        ## sanity check ##
        if x is None or z is None:
            raise ValueError(
                f"Layer {layer.name} cache is empty. Forward pass must be called "
                "before backward pass."
            )

        ## decide how to get dL/dz ##
        if (
            idx == 0
            and model.loss_function is LossFunction.BCE
            and layer.activation is ActivationFunction.SIGMOID
        ):
            """
            This is done for numerical stability, as the derivative of BCE and sigmoid
            result in a singularity when combined.
            This in essence drops the derivative of the activation function from the equation.
            As we are accounting for it in the derivative of the loss function directly.
            the derivative of the loss can be found in util.py -> bce_with_sigmoid_last_layer_grad
            """
            dL_dz: tensor_t = grad_next
        else:
            # get the derivative of the activation function #
            """
            dL_dz = dL_dy * activation_derivative(z)
            dL_dy is the gradient from the next layer
            """
            activation_derivative = ACTIVATION_DERIVATIVES[layer.activation]
            dL_dz: tensor_t = activation_derivative(z) * grad_next

        ## compute the derivative w.r.t. weights ##
        """
        dL_dw = x.T dot dL_dz / m (where m is the number of samples) 
            dimensions: (input_dim, m) dot (m, output_dim) = (input_dim, output_dim) [same as weights]
        """
        dL_dw: tensor_t = np.dot(x.T, dL_dz) / x.shape[0]

        ## compute the derivative w.r.t. bias ##
        """
        dL_db = sum(dL_dz, axis=0, keepdims=True) / m
            dimensions: (1, output_dim) [same as bias]
        """
        dL_db: tensor_t = np.sum(dL_dz, axis=0, keepdims=True) / x.shape[0]

        ## compute the derivative w.r.t. input ##
        """
        dL_dx = dL_dz dot weights.T
            dimensions: (m, output_dim) dot (output_dim, input_dim) = (m, input_dim) [same as x]
        """
        dL_dx: tensor_t = np.dot(dL_dz, layer.weights.T)

        ## store the gradients ##
        grads: Gradients = Gradients(dW=dL_dw, db=dL_db)
        grads_list.append(grads)

        ## update grad_next for the next layer ##
        grad_next = dL_dx

    ### reverse the gradients list to match the layer order ###
    grads_list.reverse()

    return grads_list


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


def train_model(
    model: Model,
    train_data: tuple[tensor_t, tensor_t],
    val_data: tuple[tensor_t, tensor_t],
    epochs: int,
    callbacks: list[Callback] = [],
) -> dict[str, list[float]]:
    """"""
    ### init history ###
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    ### unpack data ###
    tr_X, tr_y = train_data
    val_X, val_y = val_data

    ### training loop ###
    for _ in trange(epochs, desc="Training Epochs"):
        current_loss = 0.0
        for X_batch, y_batch in zip(tr_X, tr_y):
            """
            1. do a forward pass
            2. compute the loss
            3. compute the gradients
            4. apply the gradients
            5. track the loss
            """
            ### call the model ###
            y_pred = call_model(model, X_batch)

            ### compute loss ###
            loss = compute_loss(model, y_batch, y_pred)

            ### compute gradients ###
            grads = compute_gradients(model, X_batch, y_batch)

            ### apply gradients ###
            apply_gradients(model, grads)

            ### track loss ###
            current_loss += loss

        ### average loss over all batches ###
        average_loss = current_loss / len(tr_X)
        history["train_loss"].append(average_loss)

        ### validation loss ###
        y_val_pred = call_model(model, val_X)
        val_loss = compute_loss(model, val_y, y_val_pred)
        history["val_loss"].append(val_loss)

        ### Construct state for callbacks ###
        state = State(
            epoch=epochs,
            model=model,
            history=history,
            train_loss=average_loss,
            val_loss=val_loss,
        )

        ### call callbacks ###
        for callback in callbacks:
            callback(state)

        ### check for early stopping ###
        if state.stop_training:
            break

    return history


def main() -> int:
    def toy_model(x, y):
        return x ^ y

    X: tensor_t = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y: tensor_t = np.array(
        [
            [0],
            [1],
            [1],
            [0],
        ]
    )

    model: Model = create_model(
        layers=[
            create_layer(
                input_dim=(2,),
                output_dim=(3,),
                activation=ActivationFunction.RELU,
                name="layer1",
            ),
            create_layer(
                input_dim=(3,),
                output_dim=(1,),
                activation=ActivationFunction.SIGMOID,
                name="layer2",
            ),
        ],
        loss_function=LossFunction.BCE,
        learning_rate=0.1,
    )

    epochs: int = 10000
    for epoch in range(epochs):
        # Forward pass
        y_pred: tensor_t = call_model(model, X)

        # Compute loss
        loss: float = compute_loss(model, y, y_pred)

        # Compute gradients
        grads_list: list[Gradients] = compute_gradients(model, X, y)

        # Apply gradients
        apply_gradients(model, grads_list)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return 0


if __name__ == "__main__":
    exit(main())
