import shutil

from polars import DataFrame
from scipy.io import arff


def load_dataset(data_path: str) -> DataFrame:
    arff_data = arff.loadarff(data_path)
    df = DataFrame(arff_data[0])
    return df


def print_separator():
    w, h = shutil.get_terminal_size()
    print("—" * w)
