### ~~~ GLOBAL IMPORTS ~~~ ###
from matplotlib import pyplot as plt
from tqdm import trange

### ~~~ Local IMPORTS ~~~ ###
from util import (
    tensor_t,
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
    compute_gradients,
    apply_gradients,
    compute_loss,
)


### ~~~ STATE MANAGEMENT ~~~ ###
# None


def train_model(
    model: Model,
    tr_X: tensor_t,
    tr_y: tensor_t,
    val_X: tensor_t,
    val_y: tensor_t,
) -> dict[str, list[float]]:
    """"""
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    for _ in trange(EPOCHS):
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

    return history


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
        tr_X_batches,
        tr_y_batches,
        vl_X,
        vl_y,
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
