### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import Any
import numpy as np

### ~~~ Local IMPORTS ~~~ ###
from util import tensor_t, optimiser_t, Model, Gradients


def make_minibatch_sgd_optimiser(
    batch_size: int,
) -> optimiser_t:
    """
    Create a mini-batch SGD optimiser.
    Args:
        batch_size (int): The size of each mini-batch. (Not used in this function as the gradients are assumed to be pre-computed for the mini-batch)
    Returns:
        Optimiser: A mini-batch SGD optimiser function. (THIS IS CLOSURE)
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


def make_adam_optimizer(
    model: Model,
    learning_rate: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> optimiser_t:
    """
    Adam optimizer.

    learning_rate:
        If None, uses model.learning_rate.
    """
    ### init the Adam state by using Closure ###
    lr = learning_rate
    t: int = 0

    ### initialize first and second moment vectors ###
    m_w: list[tensor_t] = []
    v_w: list[tensor_t] = []
    m_b: list[tensor_t] = []
    v_b: list[tensor_t] = []

    ### for each layer, initialize m and v ###
    for layer in model.layers:
        m_w.append(np.zeros_like(layer.weights))
        v_w.append(np.zeros_like(layer.weights))
        m_b.append(np.zeros_like(layer.bias))
        v_b.append(np.zeros_like(layer.bias))

    def step(model_inner: Any, grads_list: list[Gradients]) -> None:
        """"""
        ### use nonlocal to modify t from the outer scope ###
        nonlocal t
        t += 1

        ### determine learning rate ###
        base_lr = lr if lr is not None else model_inner.learning_rate

        for i, (layer, grads) in enumerate(zip(model_inner.layers, grads_list)):
            """"""
            ## unpack gradients ##
            gW = grads.dW
            gB = grads.db

            ## update biased first moment estimate ##
            """
            m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
            """
            m_w[i] = beta1 * m_w[i] + (1 - beta1) * gW
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * gB

            ## update biased second moment estimate ##
            """
            v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t * g_t)
            """
            v_w[i] = beta2 * v_w[i] + (1 - beta2) * (gW * gW)
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gB * gB)

            ## compute bias-corrected first moment estimate ##
            """
            This is necessary because m and v are initialized as zero vectors,
            m_hat = m_i / (1 - beta_1**t)
            v_hat = v_i / (1 - beta_2**t)
            """
            m_w_hat = m_w[i] / (1 - beta1**t)
            m_b_hat = m_b[i] / (1 - beta1**t)
            v_w_hat = v_w[i] / (1 - beta2**t)
            v_b_hat = v_b[i] / (1 - beta2**t)

            ## update parameters ##
            layer.weights -= base_lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            layer.bias -= base_lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    return step
