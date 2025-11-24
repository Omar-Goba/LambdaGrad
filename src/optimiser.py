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


def _newton_schulz_orthogonalize(
    M: tensor_t,
    num_iters: int = 5,
    eps: float = 1e-8,
) -> tensor_t:
    """
    Approximate the closest orthogonal matrix to M using Newton–Schulz iteration.
    Steps:
        1. Frobenius-normalize M:
               X_0 = M / ||M||_F
        2. Iterate:
               X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
    Note:
        - This expects a 2D matrix (shape: [rows, cols]).
        - For non-2D tensors, caller should handle reshaping or skipping.
    Args:
        M (tensor_t): Input matrix to be orthogonalized.
        num_iters (int): Number of Newton–Schulz iterations to perform.
        eps (float): Small constant to avoid division by zero.
    Returns:
        tensor_t: Approximate orthogonal matrix.
    """

    ### make sure it's 2D ###
    if M.ndim != 2:
        # caller MUST decide how to handle higher dims
        # here we simply return M unchanged
        return M

    ## compute Frobenius norm ##
    """
    ||M||_F = sqrt(sum(M_ij^2))
    We add eps to avoid division by zero when M is all zeros.
    """
    frob_norm: float = (np.linalg.norm(M, ord="fro") + eps).astype(float)

    ## initialize X_0 ##
    X: tensor_t = M / frob_norm

    ## Newton–Schulz iterations ##
    for _ in range(num_iters):
        """
        X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
        """
        XtX: tensor_t = X.T @ X  # shape: (cols, cols)
        eye: tensor_t = np.eye(XtX.shape[0], dtype=X.dtype)
        X = 0.5 * X @ (3.0 * eye - XtX)

    return X


def make_muon_optimizer(
    model: Model,
    learning_rate: float | None = None,
    momentum: float = 0.95,
    num_ns_iters: int = 5,
    eps: float = 1e-8,
) -> optimiser_t:
    """
    Muon optimizer.

    High-level idea:
        1. Maintain a momentum buffer M_t for each weight matrix:
               M_t = momentum * M_{t-1} + g_t
        2. Orthogonalize M_t via Newton–Schulz iteration:
               O_t = orthogonalize(M_t)
        3. Update parameters with orthogonalized step:
               W <- W - lr * sigma * O_t
           where sigma is a scaling factor based on the matrix shape.

    Notes:
        - Muon is defined for 2D matrices (e.g., linear layer weights with shape [in_dim, out_dim]).
        - Biases (1D vectors) are typically updated with standard momentum SGD and not orthogonalized.
        - For conv kernels etc., one would reshape to 2D, apply NS, then reshape back.
          Here we assume simple 2D dense layers.

    Args:
        model (Model):
            The model whose parameters we will optimize.
        learning_rate (float | None):
            If None → use model.learning_rate on each step.
        momentum (float):
            Momentum factor for the running gradient buffer (e.g., 0.95).
        num_ns_iters (int):
            Number of Newton–Schulz refinement iterations.
        eps (float):
            Small constant for numerical stability (norms, etc.).

    Returns:
        optimiser_t:
            A closure `step(model, grads_list)` that performs one Muon update.
    """

    ### init the Muon state by using Closure ###
    lr = learning_rate

    ### momentum buffers for weights and biases ###
    M_w: list[tensor_t] = []
    M_b: list[tensor_t] = []

    ### for each layer, initialize momentum buffers ###
    for layer in model.layers:
        M_w.append(np.zeros_like(layer.weights))
        M_b.append(np.zeros_like(layer.bias))

    def step(model_inner: Any, grads_list: list[Gradients]) -> None:
        """"""
        ### determine learning rate ###
        base_lr = lr if lr is not None else model_inner.learning_rate

        for i, (layer, grads) in enumerate(zip(model_inner.layers, grads_list)):
            """"""
            ## unpack gradients ##
            gW: tensor_t = grads.dW
            gB: tensor_t = grads.db

            ## update momentum buffer for weights ##
            """
            M_t = momentum * M_{t-1} + g_t
            """
            M_w[i] = momentum * M_w[i] + gW

            ## update momentum buffer for biases (no orthogonalization) ##
            """
            For biases (1D), Muon typically does not apply the orthogonalization step.
            We just use classic momentum SGD for them.
            """
            M_b[i] = momentum * M_b[i] + gB

            ## orthogonalize momentum for weights using Newton–Schulz ##
            """
            Only valid for 2D matrices.
            If the weight tensor is not 2D, we skip orthogonalization and fall back to
            using the raw momentum buffer.
            """
            if layer.weights.ndim == 2:
                O_w: tensor_t = _newton_schulz_orthogonalize(
                    M_w[i],
                    num_iters=num_ns_iters,
                    eps=eps,
                )
            else:
                # Fallback: no orthogonalization (e.g., for non-2D tensors)
                O_w = M_w[i]

            ## compute scaling factor sigma based on matrix shape ##
            """
            Common choice:
                sigma = max(1, sqrt(rows / cols))

            This keeps the variance of the update under control, especially for
            very tall or very wide matrices. Look into Random Matrix Theory for the proof.
            """
            if layer.weights.ndim == 2:
                rows, cols = layer.weights.shape
                sigma = max(1.0, np.sqrt(rows / max(cols, 1)))
            else:
                sigma = 1.0  # safe default

            ## update weights with orthogonalized step ##
            """
            W <- W - lr * sigma * O_w
            """
            layer.weights -= base_lr * sigma * O_w

            ## update biases with standard momentum SGD ##
            """
            b <- b - lr * M_b
            """
            layer.bias -= base_lr * M_b[i]

    return step
