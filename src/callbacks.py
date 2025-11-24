### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import Callable, Any
from dataclasses import dataclass

### ~~~ LOCAL IMPORTS ~~~ ###


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass
class State:
    epoch: int
    model: Any
    history: dict[str, list[float]]
    train_loss: float
    val_loss: float
    stop_training: bool = False


Callback = Callable[[State], None]


### ~~~ STATE MANAGEMENT ~~~ ###
# None


def make_early_stopping_callback(
    patience: int = 5,
    min_delta: float = 0.0,
) -> Callback:
    """
    Create an early stopping callback based on validation loss.
    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
    Returns:
        Callback: A callback function for early stopping. (THIS IS CLOSURE)
    """

    best_val_loss: float = float("inf")
    epochs_without_improve: int = 0

    def callback(state: State) -> None:
        nonlocal best_val_loss, epochs_without_improve

        current_val_loss = state.val_loss

        ### check improvement ###
        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        ### trigger stop if patience exhausted ###
        if epochs_without_improve >= patience:
            print(
                f"Early stopping: no improvement in val_loss for {patience} epochs. "
                f"Best val_loss={best_val_loss:.4f}"
            )
            state.stop_training = True

    return callback


def make_lr_annealing_callback(
    factor: float = 0.5,
    patience: int = 5,
    min_delta: float = 0.0,
    min_lr: float = 1e-5,
) -> Callback:
    """
    Reduce learning rate by `factor` if validation loss does not improve
    for `patience` epochs.
    Args:
        factor (float): Factor to reduce learning rate by.
        patience (int): Number of epochs to wait for improvement before reducing LR.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
        min_lr (float): Minimum learning rate allowed.
    Returns:
        Callback: A callback function for learning rate annealing. (THIS IS CLOSURE)
    """

    best_val_loss: float = float("inf")
    epochs_without_improve: int = 0

    def callback(state: State) -> None:
        nonlocal best_val_loss, epochs_without_improve

        model: Any = state.model
        current_val_loss = state.val_loss

        ### check improvement ###
        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            epochs_without_improve = 0
            return

        epochs_without_improve += 1

        ### update LR if plateau ###
        if epochs_without_improve >= patience:
            old_lr = model.learning_rate
            new_lr = max(old_lr * factor, min_lr)

            if new_lr < old_lr:
                print(
                    f"LR annealing: val_loss plateaued for {patience} epochs. "
                    f"LR {old_lr:.5f} -> {new_lr:.5f}"
                )
                model.learning_rate = new_lr

            epochs_without_improve = 0  # reset patience after LR change

    return callback
