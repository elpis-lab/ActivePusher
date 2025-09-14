import os, sys
import warnings
import numpy as np
import torch
import argparse

# for better printing
warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(precision=5, suppress=True)


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_args(args):
    """
    A simple wrapper for argument parser
    args is a list of arguments, each argument is
    a tuple of (name, default(optional), type(optional))
    """
    parser = argparse.ArgumentParser()
    for arg in args:
        kwargs = {"nargs": "?"}
        if len(arg) > 1:
            kwargs["default"] = arg[1]
        if len(arg) > 2:
            kwargs["type"] = arg[2]
        parser.add_argument(arg[0], **kwargs)

    args = parser.parse_args()
    return args


def get_names(object_name):
    """A simple function to extract where the model and data name should be"""
    if "real" in object_name:
        model_name = object_name[5:]
        data_name = object_name + "_2000"  # real world data has 2000 samples
    else:
        model_name = object_name
        data_name = object_name + "_10000"  # sim data has 10000 samples
    return model_name, data_name


class DataLoader:
    """Class for loading data and splitting them"""

    def __init__(
        self,
        data_name: str,
        val_size: int = 0,
        test_size: int = 0,
        folder: str = "data",
        invert_xy: bool = False,
        shuffle: bool = False,
    ):
        """Initialize with data files, split sizes, and some options"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder = os.path.join(root, folder)
        self.data_x_file = folder + "/x_" + data_name + ".npy"
        self.data_y_file = folder + "/y_" + data_name + ".npy"
        self.val_size = val_size
        self.test_size = test_size
        # Load data (convert to float32 for torch)
        x = np.load(self.data_x_file).astype(np.float32)
        y = np.load(self.data_y_file).astype(np.float32)

        # Check
        assert len(x) == len(y), "Data x and y should have the same length"
        assert self.val_size >= 0, "Validation size must be non-negative"
        assert self.test_size >= 0, "Test size must be non-negative"
        self.pool_size = len(x) - self.val_size - self.test_size
        if self.pool_size < 0:
            raise ValueError("Pool size is negative")

        # Pre-process data
        if invert_xy:
            x, y = y, x
        if shuffle:
            idx = np.random.permutation(len(x))
            x, y = x[idx], y[idx]
        self.x = x
        self.y = y

    def load_data(self, verbose=0):
        """Load all data as a dictionary"""
        # Split data
        x_pool = self.x[: self.pool_size]
        y_pool = self.y[: self.pool_size]
        x_val = self.x[self.pool_size : self.pool_size + self.val_size]
        y_val = self.y[self.pool_size : self.pool_size + self.val_size]
        x_test = self.x[self.pool_size + self.val_size :]
        y_test = self.y[self.pool_size + self.val_size :]
        if verbose:
            print("Loading data")
            print(f"Pool data points: {x_pool.shape[0]}")
            print(f"Validation data points: {x_val.shape[0]}")
            print(f"Test data points: {x_test.shape[0]}")

        datasets = dict()
        datasets["x_pool"] = x_pool
        datasets["y_pool"] = y_pool
        datasets["x_val"] = x_val
        datasets["y_val"] = y_val
        datasets["x_test"] = x_test
        datasets["y_test"] = y_test
        return datasets
