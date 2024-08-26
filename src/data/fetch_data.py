import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
)  # we use stratified kfold, because we have imbalanced classes and stratified means we keep the same proportions of classes (0,1) in each fold
from tqdm.auto import tqdm

tqdm.pandas()


def fetch_data(input_path: str, only_malignant: str, save: bool) -> pd.DataFrame:
    """
    Fetches the data from the given path and returns a pandas dataframe
    """
    print("Fetching data...")

    DATA_2024_PATH = f"{input_path}/isic-2024-challenge"
    DATA_2020_PATH = f"{input_path}/isic-2020-jpg-256x256"
    DATA_2019_PATH = f"{input_path}/isic-2019-jpg-256x256"
    DATA_2018_PATH = f"{input_path}/isic-2018-jpg-256x256"
    DATA_2016_PATH = f"{input_path}/isic-2016"

    print(f"datapath {DATA_2024_PATH}")

    df_train_2024 = pd.read_csv(f"{DATA_2024_PATH}/train-metadata.csv")
    df_train_2020 = pd.read_csv(f"{DATA_2020_PATH}/train-metadata.csv")
    df_train_2019 = pd.read_csv(f"{DATA_2019_PATH}/train-metadata.csv")
    df_train_2018 = pd.read_csv(f"{DATA_2018_PATH}/train-metadata.csv")
    df_train_2016 = pd.read_csv(f"{DATA_2016_PATH}/train-metadata.csv")

    # first 2016 df
    df_train_2016.columns = ["isic_id", "target"]
    df_train_2016["target"] = df_train_2016["target"].progress_apply(lambda x: 1 if x == "malignant" else 0)

    # fix 2018 target to int
    df_train_2018["target"] = df_train_2018.target.astype(int)

    # only have 2 cols: isic_id, target
    cols = ["isic_id", "target"]

    df_train_2018 = df_train_2018[cols]
    df_train_2019 = df_train_2019[cols]
    df_train_2020 = df_train_2020[cols]
    df_train_2024 = df_train_2024[cols]

    check_path = lambda p: os.path.exists(p)

    get_image_path_2024 = lambda p: os.path.join(f"{DATA_2024_PATH}/train-image/image/{p}.jpg")
    get_image_path_2020 = lambda p: os.path.join(f"{DATA_2020_PATH}/train-image/image/{p}.jpg")
    get_image_path_2019 = lambda p: os.path.join(f"{DATA_2019_PATH}/train-image/image/{p}.jpg")
    get_image_path_2018 = lambda p: os.path.join(f"{DATA_2018_PATH}/train-image/image/{p}.jpg")
    get_image_path_2016 = lambda p: os.path.join(f"{DATA_2016_PATH}/train-image/image/{p}.jpg")

    df_train_2024["image_path"] = df_train_2024["isic_id"].progress_apply(get_image_path_2024)
    df_train_2020["image_path"] = df_train_2020["isic_id"].progress_apply(get_image_path_2020)
    df_train_2019["image_path"] = df_train_2019["isic_id"].progress_apply(get_image_path_2019)
    df_train_2018["image_path"] = df_train_2018["isic_id"].progress_apply(get_image_path_2018)
    df_train_2016["image_path"] = df_train_2016["isic_id"].progress_apply(get_image_path_2016)

    df_train_2024["image_exists"] = df_train_2024["image_path"].progress_apply(check_path)
    df_train_2020["image_exists"] = df_train_2020["image_path"].progress_apply(check_path)
    df_train_2019["image_exists"] = df_train_2019["image_path"].progress_apply(check_path)
    df_train_2018["image_exists"] = df_train_2018["image_path"].progress_apply(check_path)
    df_train_2016["image_exists"] = df_train_2016["image_path"].progress_apply(check_path)

    # remove datapoints where image does not exist
    df_train_2024 = df_train_2024[df_train_2024["image_exists"]][["isic_id", "target", "image_path"]]
    df_train_2020 = df_train_2020[df_train_2020["image_exists"]][["isic_id", "target", "image_path"]]
    df_train_2019 = df_train_2019[df_train_2019["image_exists"]][["isic_id", "target", "image_path"]]
    df_train_2018 = df_train_2018[df_train_2018["image_exists"]][["isic_id", "target", "image_path"]]
    df_train_2016 = df_train_2016[df_train_2016["image_exists"]][["isic_id", "target", "image_path"]]

    if only_malignant:
        df_train_2020 = df_train_2020[df_train_2020["target"] == 1]
        df_train_2019 = df_train_2019[df_train_2019["target"] == 1]
        df_train_2018 = df_train_2018[df_train_2018["target"] == 1]
        df_train_2016 = df_train_2016[df_train_2016["target"] == 1]

    df_train_2024["dataset"] = "2024"
    df_train_2020["dataset"] = "2020"
    df_train_2019["dataset"] = "2019"
    df_train_2018["dataset"] = "2018"
    df_train_2016["dataset"] = "2016"

    df = pd.concat([df_train_2024, df_train_2020, df_train_2019, df_train_2018, df_train_2016])
    print(f"{'=' * 20} Final dataset {'=' * 20}")
    print(f"Number of malignant samples: {df[df['target'] == 1].shape[0]}")
    print(f"Number of benign samples: {df[df['target'] == 0].shape[0]}")
    if save:
        df.to_csv(f"{input_path}/train_metadata_combined.csv", index=False)

    return df


def create_folds(data: pd.DataFrame, n_splits: int, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Creates k-folds from the given data
    """
    print(f"columns: {data.columns}")
    data["fold"] = -1

    print(f"columns: {data.columns}")

    strat_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Split the data and assign fold numbers
    for fold, (train_idx, val_idx) in enumerate(strat_kfold.split(data, data["target"])):
        data.loc[val_idx, "fold"] = fold

    return data


if __name__ == "__main__":
    cwd = os.getcwd()
    input_path = "kaggle/input"
    input_path = os.path.join(cwd, input_path)
    remake_file = False
    if "train_metadata_combined.csv" in os.listdir(input_path) and not remake_file:
        print("Metadata file exists")
        df = pd.read_csv(f"{input_path}/train_metadata_combined.csv")
    else:
        df = fetch_data(input_path, only_malignant=False, save=True)

    print("Creating k-folds...")
    df_kfold = create_folds(df, n_splits=5, random_state=42)
    df_kfold.to_csv(f"{input_path}/train_metadata_combined_kfold.csv", index=False)

    print("Creating pure malignant, pure benign and mix5050 datasets...")
    pure_malignant = df[df["target"] == 1]
    pure_benign = df[df["target"] == 0]
    pure_malignant.to_csv(f"{input_path}/pure_malignant.csv", index=False)
    pure_benign.to_csv(f"{input_path}/pure_benign.csv", index=False)
    mix5050 = pd.concat([pure_malignant, pure_benign.sample(n=pure_malignant.shape[0], random_state=42)])
    mix5050.drop(columns=["fold"], inplace=True)
    # shuffle the mix5050 dataset, to make sure the malignant and benign samples are mixed
    mix5050 = mix5050.sample(frac=1, random_state=42)
    mix5050.reset_index(drop=True, inplace=True)

    # Debugging: Check if 'target' column exists and inspect the DataFrame
    print("mix5050 columns:", mix5050.columns)
    print("mix5050 sample:", mix5050.head())

    # Ensure 'target' column is still present
    if "target" not in mix5050.columns:
        raise ValueError("Target column is missing from mix5050 dataset.")

    # Create folds for the mix5050 dataset
    mix5050kfold = create_folds(mix5050, n_splits=5, random_state=42)

    # Debugging: Check the resulting DataFrame
    print("mix5050kfold sample:", mix5050kfold.head())

    mix5050kfold.to_csv(f"{input_path}/mix5050.csv", index=False)
