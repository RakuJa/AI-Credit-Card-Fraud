import os
from pathlib import Path

import onnxmltools.convert.xgboost
from onnxconverter_common import FloatTensorType
from xgboost import XGBClassifier
from polars import DataFrame

import data_parser
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
    save_graphs = True
    show_graphs = False
    raw_data: DataFrame = load_dataset("dataset/data.arff")
    if os.getenv("EXPLORE_DATASET", "TRUE").upper() == "TRUE":
        _df = explore_dataset(
            df=raw_data.clone(), show_graphs=show_graphs, save_graphs=save_graphs
        )
    data_parser.check_validation_method(
        raw_data, show_graphs=show_graphs, save_graphs=save_graphs
    )
    df, x_train_smt, y_train_smt, x_test, y_test = prepare_dataset(
        df=raw_data, show_graphs=show_graphs, save_graphs=save_graphs
    )

    if os.getenv("EXPLORE_MODELS", "TRUE").upper() == "TRUE":
        explore_models(
            x_train_smt,
            y_train_smt,
            x_test,
            y_test,
            show_graphs=show_graphs,
            save_graphs=save_graphs,
        )
    execute_chosen_model(df, x_test, y_test, x_train_smt, y_train_smt)


def execute_chosen_model(df, x_test, y_test, x_train_smt, y_train_smt):
    json_path: str = "model/xgb.json"
    if Path(json_path).exists():
        xgb_opt: XGBClassifier = XGBClassifier()
        xgb_opt.load_model(json_path)
    else:
        xgb_opt: XGBClassifier = execute(x_train_smt, y_train_smt, x_test, y_test)
        xgb_opt.save_model(json_path)
    save_to_onnx(xgb_opt, "model/model.onnx")

    model_performance(xgb_opt, df, x_test, y_test)


def save_to_onnx(xgb_opt: XGBClassifier, file_path: str):
    onnx_model = onnxmltools.convert.xgboost.convert(
        model=xgb_opt,
        initial_types=[("input", FloatTensorType([None, 30]))],
        target_opset=15,
    )

    onnxmltools.save_model(onnx_model, file_path)


if __name__ == "__main__":
    main()
