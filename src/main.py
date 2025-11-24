### ~~~ GLOBAL IMPORTS ~~~ ###
from matplotlib import pyplot as plt

### ~~~ Local IMPORTS ~~~ ###
from util import (
    load_data,
    RATIOS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    ActivationFunction,
    LossFunction,
)
from explore import feature_engineering
from process import impute_data, split, batch_data

from model import (
    Model,
    call_model,
    create_layer,
    compute_loss,
    train_model,
)


### ~~~ STATE MANAGEMENT ~~~ ###
# None


def main() -> int:
    """"""
    ### load the data ###
    df = load_data()

    ### impute the data ###
    df = impute_data(df)

    ### feature engineering ###
    # feature_engineering(df)

    ### split the data ###
    splits = split(df, RATIOS)
    tr_X, tr_y = splits["train"]
    vl_X, vl_y = splits["validation"]
    ts_X, ts_y = splits["test"]

    ### batch the data ###
    tr_X_batches, tr_y_batches = batch_data(tr_X, tr_y, BATCH_SIZE)

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
    )

    ### train the model ###
    history = train_model(
        model,
        (tr_X_batches, tr_y_batches),
        (vl_X, vl_y),
        epochs=EPOCHS,
    )

    ### plot the loss curves ###
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.show()

    y_test_pred = call_model(model, ts_X)
    test_loss = compute_loss(model, ts_y, y_test_pred)
    print(f"Test Loss: {test_loss}")

    return 0


if __name__ == "__main__":
    exit(main())
