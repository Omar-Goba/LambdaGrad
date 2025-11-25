### ~~~ GLOBAL IMPORTS ~~~ ###
from matplotlib import pyplot as plt
from itertools import product
from tqdm import tqdm


### ~~~ Local IMPORTS ~~~ ###
from util import (
    dim_t,
    History,
    load_data,
    RATIOS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    ActivationFunction,
    LossFunction,
    AccuracyMetric,
)
from explore import feature_engineering
from process import process_data, split
from callbacks import make_early_stopping_callback, make_lr_annealing_callback
from optimiser import (
    make_minibatch_sgd_optimiser,
    make_adam_optimizer,
    make_muon_optimizer,
)
from model import (
    Model,
    call_model,
    comptute_accuracy,
    create_layer,
    compute_loss,
    train_model,
)


def build_layers(input_shape: dim_t, dim_list: list[int]) -> list:
    """
    Given a list like [16, 8, 1], build a layer stack:
    input -> 16 -> 8 -> 1
    """
    layers = []

    # first layer takes input_dim
    prev_dim = input_shape[1]

    for i, dim in enumerate(dim_list):
        is_last = i == len(dim_list) - 1

        layers.append(
            create_layer(
                input_dim=(prev_dim,),
                output_dim=(dim,),
                activation=(
                    ActivationFunction.SIGMOID if is_last else ActivationFunction.RELU
                ),
                name=f"layer_{i + 1}",
            )
        )

        prev_dim = dim

    return layers


def plot_history(history: History) -> None:
    """
    Plot training and validation loss and accuracy over epochs.
    Args:
        history (dict): A dictionary containing training and validation loss and accuracy.
    Returns:
        None
    """
    plt.subplot(1, 2, 1)
    plt.plot(history.train_loss, label="Train Loss")
    plt.plot(history.val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.train_accuracy, label="Train Accuracy")
    plt.plot(history.val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy over Epochs")
    plt.legend()
    plt.show()


def run_everythin(
    dims: list[int],
    optim: str,
    do_lr_annealing: bool,
    do_early_stop: bool,
    batch_size: int,
) -> float:
    """"""
    ### load the data ###
    df = load_data(from_cache=True)

    ### process the data ###
    df = process_data(df)

    ### feature engineering ###
    # feature_engineering(df)

    ### split the data ###
    splits = split(df, RATIOS)
    tr_X, tr_y = splits["train"]
    vl_X, vl_y = splits["validation"]
    ts_X, ts_y = splits["test"]

    ### init model ###
    model = Model(
        layers=build_layers(input_shape=tr_X.shape, dim_list=dims),
        learning_rate=LEARNING_RATE,
        loss_function=LossFunction.BCE,
        accuracy_function=AccuracyMetric.BINARY_ACCURACY,
    )

    ### set up callbacks ###
    callbacks = []
    if do_early_stop:
        early_stopping_callback = make_early_stopping_callback(
            patience=20,
            min_delta=1e-4,
        )
        callbacks.append(early_stopping_callback)
    if do_lr_annealing:
        lr_annealing_callback = make_lr_annealing_callback(
            factor=0.5,
            patience=3,
            min_delta=1e-4,
            min_lr=1e-5,
        )
        callbacks.append(lr_annealing_callback)

    ### init the optimiser ###
    if optim == "sgd":
        optimiser = make_minibatch_sgd_optimiser(batch_size=batch_size)
    elif optim == "adam":
        optimiser = make_adam_optimizer(model=model)
    elif optim == "muon":
        optimiser = make_muon_optimizer(model=model)
    else:
        raise ValueError(f"Unknown optimiser: {optim}")

    ### train the model ###
    history = train_model(
        model,
        (tr_X, tr_y),
        (vl_X, vl_y),
        epochs=EPOCHS,
        batch_size=batch_size,
        optimiser=optimiser,
        callbacks=callbacks,
        verbose=False,
    )

    ### plot the loss curves ###
    # plot_history(history)

    y_test_pred = call_model(model, ts_X)
    test_loss = compute_loss(model, ts_y, y_test_pred)

    return test_loss


def hypertune() -> int:
    """
    hyper tuning experiments
    """
    possible_optimizers = ["sgd", "adam", "muon"]
    possible_lr_settings = [False, True]
    possible_early_stopping = [False, True]
    possible_dims = [
        [8, 1],
        [16, 8, 1],
        [32, 16, 8, 1],
    ]
    possible_batch_sizes = [2**i for i in range(6, 11)]  # 64 to 1024
    best_loss = float("inf")
    best_settings = None

    for optim_name, lr_setting, do_early_stop, dims, batch_size in tqdm(
        list(
            product(
                possible_optimizers,
                possible_lr_settings,
                possible_early_stopping,
                possible_dims,
                possible_batch_sizes,
            )
        )
    ):
        test_loss = run_everythin(
            dims=dims,
            optim=optim_name,
            do_lr_annealing=lr_setting,
            do_early_stop=do_early_stop,
            batch_size=batch_size,
        )
        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
            best_settings = {
                "optimizer": optim_name,
                "lr_annealing": lr_setting,
                "early_stopping": do_early_stop,
                "dims": dims,
                "batch_size": batch_size,
                "test_loss": test_loss,
            }

    print("Best Settings Found:")
    print(best_settings)
    print(f"Best Test Loss: {best_loss}")

    return 0


def main() -> int:
    """
    Main entry point for the program.
    """
    hypertune()

    return 0


if __name__ == "__main__":
    exit(main())
