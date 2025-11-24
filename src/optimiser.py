### ~~~ GLOBAL IMPORTS ~~~ ###
# None

### ~~~ Local IMPORTS ~~~ ###
from util import optimiser_t, Model, Gradients


def make_minibatch_sgd_optimizer(
    batch_size: int,
) -> optimiser_t:
    """
    Create a mini-batch SGD optimizer.
    Args:
        batch_size (int): The size of each mini-batch.
    Returns:
        Optimizer: A mini-batch SGD optimizer function. (THIS IS CLOSURE)
    """

    def step(model: Model, gradients_list: list[Gradients]) -> None:
        """
        Perform a single optimization step using mini-batch SGD.
        Args:
            model (Model): The neural network model.
            gradients_list (list[Gradients]): List of gradients for each layer.
        Returns:
            None
        """
        for layer, gradients in zip(model.layers, gradients_list):
            layer.weights -= model.learning_rate * gradients.dW
            layer.bias -= model.learning_rate * gradients.db

    return step
