### ~~~ GLOBAL IMPORTS ~~~ ###
# None

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
    compute_gradients,
    apply_gradients,
    compute_loss,
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
    history = []
    for epoch in range(EPOCHS):
        current_loss = 0.0
        for X_batch, y_batch in zip(tr_X_batches, tr_y_batches):
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

        average_loss = current_loss / len(tr_X_batches)
        history.append(average_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
