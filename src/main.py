### ~~~ GLOBAL IMPORTS ~~~ ###
# None

### ~~~ Local IMPORTS ~~~ ###
from util import load_data, RATIOS
from explore import feature_engineering
from process import impute_data, split


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

    return 0


if __name__ == "__main__":
    exit(main())
