import os
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from polars import DataFrame

from data_explorer import explore_dataset
from fine_tuning import execute, model_performance
from model_explorer import (
    explore_models,
)
from data_parser import prepare_dataset
from utils import load_dataset
from dotenv import load_dotenv


def main():
    load_dotenv()  # take environment variables

    raw_data: DataFrame = load_dataset("dataset/data.arff")
    if os.getenv("EXPLORE_DATASET", "TRUE").upper() == "TRUE":
        _df = explore_dataset(df=raw_data.clone(), show_graphs=False, save_graphs=True)
    df, x_train_smt, y_train_smt, x_test, y_test = prepare_dataset(df=raw_data)
    if os.getenv("EXPLORE_MODELS", "TRUE").upper() == "TRUE":
        explore_models(
            x_train_smt,
            y_train_smt,
            x_test,
            y_test,
            show_graphs=False,
            save_graphs=True,
        )
    if Path("model/lgb.pkl").exists():
        lgb_opt: LGBMClassifier = joblib.load("model/lgb.pkl")
    else:
        lgb_opt: LGBMClassifier = execute(x_train_smt, y_train_smt, x_test, y_test)
        joblib.dump(lgb_opt, "model/lgb.pkl")
        lgb_opt.booster_.save_model("model/mode.txt")
    model_performance(lgb_opt, df, x_test, y_test)


if __name__ == "__main__":
    main()
