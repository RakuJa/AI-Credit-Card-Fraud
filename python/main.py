import os
from pathlib import Path

import joblib
import numpy as np
import onnxmltools.convert.xgboost
from onnxconverter_common import FloatTensorType
from xgboost import XGBClassifier
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
    if Path("model/xgb.json").exists():
        xgb_opt: XGBClassifier = XGBClassifier()
        xgb_opt.load_model("model/xgb.json")
    else:
        xgb_opt: XGBClassifier = execute(x_train_smt, y_train_smt, x_test, y_test)
        joblib.dump(xgb_opt, "model/xgb.pkl")
        xgb_opt.save_model("model/xgb.json")
    save_to_onnx(xgb_opt, "model/model.onnx")

    model_performance(xgb_opt, df, x_test, y_test)

def save_to_onnx(xgb_opt: XGBClassifier, file_path: str):
    onnx_model = onnxmltools.convert.xgboost.convert(model=xgb_opt, initial_types=[
        ("input", FloatTensorType([None, 30]))
    ],target_opset=15)

    onnxmltools.save_model(onnx_model, file_path)


if __name__ == "__main__":
    main()
