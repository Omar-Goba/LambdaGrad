### ~~~ GLOBAL IMPORTS ~~~ ###
# None

### ~~~ Local IMPORTS ~~~ ###
from util import load_data, RATIOS, BATCH_SIZE, EPOCHS, tensor_t
from explore import feature_engineering
from process import impute_data, split, batch_data


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

    return 0


if __name__ == "__main__":
    exit(main())
