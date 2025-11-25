### ~~~ GLOBAL IMPORTS ~~~ ###
from matplotlib import pyplot as plt

### ~~~ Local IMPORTS ~~~ ###
from util import (
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
    create_layer,
    compute_loss,
    train_model,
)


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


def main() -> int:
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
        layers=[
            create_layer(
                input_dim=(tr_X.shape[1],),
                output_dim=(16,),
                activation=ActivationFunction.RELU,
                name="hidden_1",
            ),
            create_layer(
                input_dim=(16,),
                output_dim=(8,),
                activation=ActivationFunction.RELU,
                name="hidden_2",
            ),
            create_layer(
                input_dim=(8,),
                output_dim=(1,),
                activation=ActivationFunction.SIGMOID,
                name="output",
            ),
        ],
        learning_rate=LEARNING_RATE,
        loss_function=LossFunction.BCE,
        accuracy_function=AccuracyMetric.BINARY_ACCURACY,
    )

    ### init optimiser ###
    sgd_optim = make_minibatch_sgd_optimiser(batch_size=BATCH_SIZE)
    adam_optim = make_adam_optimizer(model=model)
    muon_optim = make_muon_optimizer(model=model)
    optimiser = adam_optim

    ### set up callbacks ###
    early_stopping_callback = make_early_stopping_callback(
        patience=20,
        min_delta=1e-4,
    )
    lr_annealing_callback = make_lr_annealing_callback(
        factor=0.5,
        patience=3,
        min_delta=1e-4,
        min_lr=1e-5,
    )

    ### train the model ###
    history = train_model(
        model,
        (tr_X, tr_y),
        (vl_X, vl_y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        optimiser=optimiser,
        callbacks=[early_stopping_callback, lr_annealing_callback],
    )

    ### plot the loss curves ###
    plot_history(history)

    y_test_pred = call_model(model, ts_X)
    test_loss = compute_loss(model, ts_y, y_test_pred)
    print(f"Test Loss: {test_loss}")

    """
    experiments:
      - optimizer: SGD
        results:
          - lr_setting: 0
            early_stopping: 0
            test_loss: 0.2479
            is_best: false
          - lr_setting: 1
            early_stopping: 0
            test_loss: 0.2247  # Overall Best Result
            is_best: true
          - lr_setting: 0
            early_stopping: 1
            test_loss: 0.2720
            is_best: false
          - lr_setting: 1
            early_stopping: 1
            test_loss: 0.2253
            is_best: false

      - optimizer: Adam
        results:
          - lr_setting: 0
            early_stopping: 0
            test_loss: 0.6723  # Best Adam Result
            is_best: true
          - lr_setting: 1
            early_stopping: 0
            test_loss: 0.6724
            is_best: false
          - lr_setting: 0
            early_stopping: 1
            test_loss: 0.6726
            is_best: false
          - lr_setting: 1
            early_stopping: 1
            test_loss: 0.6735
            is_best: false

      - optimizer: Muon
        results:
          - lr_setting: 0
            early_stopping: 0
            test_loss: 0.6724
            is_best: false
          - lr_setting: 1
            early_stopping: 0
            test_loss: 0.6712
            is_best: false
          - lr_setting: 0
            early_stopping: 1
            test_loss: 0.6719
            is_best: false
          - lr_setting: 1
            early_stopping: 1
            test_loss: 0.6712  # Best Muon Result (Tied)
            is_best: true

    best_overall_configuration:
      optimizer: SGD
      lr_setting: 1
      early_stopping: 0
      test_loss: 0.2247
    """

    return 0


if __name__ == "__main__":
    exit(main())
