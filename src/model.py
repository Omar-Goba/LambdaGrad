### ~~~ GLOBAL IMPORTS ~~~ ###
from tqdm import trange
import numpy as np

### ~~~ Local IMPORTS ~~~ ###
from process import batch_data
from util import (
    RNG,
    tensor_t,
    optimiser_t,
    LossFunction,
    LOSS_FUNCTIONS,
    LOSS_DERIVATIVES,
    ActivationFunction,
    ACTIVATION_DERIVATIVES,
    AccuracyMetric,
    ACCURACY_METRICS,
    Model,
    History,
)
from layer import Layer, Gradients, call_layer, layer_to_str, create_layer
from callbacks import callback_t, State


def create_model(
    layers: list[Layer],
    loss_function: LossFunction,
    accuracy_function: AccuracyMetric = AccuracyMetric.BINARY_ACCURACY,
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
        accuracy_function=accuracy_function,
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


def comptute_accuracy(model: Model, y_true: tensor_t, y_pred: tensor_t) -> float:
    """
    Compute the accuracy of the model.
    Args:
        model (Model): The neural network model.
        y_true (tensor_t): The true labels.
        y_pred (tensor_t): The predicted labels.
    Returns:
        float: The computed accuracy.
    """
    accuracy_function = ACCURACY_METRICS[model.accuracy_function]
    accuracy: float = accuracy_function(y_true, y_pred)

    return accuracy


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
    batch_size: int,
    optimiser: optimiser_t,
    callbacks: list[callback_t] = [],
    do_shuffle: bool = True,
    verbose: bool = True,
) -> History:
    """
    This function trains the model using whatever optimiser is passed to it.
    It does the following:
    - Initialises the history object to track training and validation loss and accuracy.
    - Shuffles the training data if specified.
    - Batches the training data.
    - For each epoch:
        - For each batch:
            - Performs a forward pass.
            - Computes the loss and accuracy.
            - Computes the gradients.
            - Applies the gradients using the optimiser.
            - Updates the history.
        - Computes validation loss and accuracy.
        - Calls any callbacks with the current state.
        - Checks for early stopping.
    Args:
        model (Model): The neural network model to be trained.
        train_data (tuple[tensor_t, tensor_t]): The training data (features and labels).
        val_data (tuple[tensor_t, tensor_t]): The validation data (features and labels).
        epochs (int): The number of epochs to train the model.
        batch_size (int): The size of each training batch.
        optimiser (optimiser_t): The optimiser function to apply gradients.
        callbacks (list[callback_t], optional): List of callback functions to be called
            at the end of each epoch. Defaults to [].
        do_shuffle (bool, optional): Whether to shuffle the training data before each epoch.
            Defaults to True.
    Returns:
        History: The training history containing loss and accuracy for training and validation.
    """
    ### init history ###
    history: History = History(
        train_loss=[],
        val_loss=[],
        train_accuracy=[],
        val_accuracy=[],
    )

    ### unpack data ###
    tr_X, tr_y = train_data
    val_X, val_y = val_data

    ### shuffle training data ###
    if do_shuffle:
        perm: np.ndarray = RNG.permutation(tr_X.shape[0])
        tr_X = tr_X[perm]
        tr_y = tr_y[perm]

    ### batch training data ###
    tr_X_batches, tr_y_batches = batch_data(tr_X, tr_y, batch_size)

    ### training loop ###
    for _ in trange(epochs, desc="Training Epochs", disable=not verbose):
        current_loss = 0.0
        current_accuracy = 0.0
        for X_batch, y_batch in zip(tr_X_batches, tr_y_batches):
            """
            1. do a forward pass
            2. compute the loss
            3. compute the accuracy
            4. compute the gradients
            5. apply the gradients
            6. track the history
            """
            ### call the model ###
            y_pred = call_model(model, X_batch)

            ### compute loss ###
            loss = compute_loss(model, y_batch, y_pred)

            ### compute accuracy ###
            accuracy = comptute_accuracy(model, y_batch, y_pred)

            ### compute gradients ###
            grads = compute_gradients(model, X_batch, y_batch)

            ### apply gradients ###
            optimiser(model, grads)

            ### track history ###
            current_loss += loss
            current_accuracy += accuracy

        ### average history over all batches ###
        average_loss = current_loss / len(tr_X)
        history.train_loss.append(average_loss)
        average_accuracy = current_accuracy / len(tr_X)
        history.train_accuracy.append(average_accuracy)

        ### validation loss ###
        y_val_pred = call_model(model, val_X)
        val_loss = compute_loss(model, val_y, y_val_pred)
        history.val_loss.append(val_loss)
        val_accuracy = comptute_accuracy(model, val_y, y_val_pred)
        history.val_accuracy.append(val_accuracy)

        ### Construct state for callbacks ###
        state = State(
            epoch=epochs,
            model=model,
            history=history,
            train_loss=average_loss,
            val_loss=val_loss,
        )

        ### call callbacks ###
        for cb in callbacks:
            cb(state)

        ### check for early stopping ###
        if state.stop_training:
            break

    return history


if __name__ == "__main__":
    ...
