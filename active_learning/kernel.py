import os
import sys
import torch
import torch.nn as nn

from bmdal_reg.bmdal.algorithms import select_batch
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.layers import (
    LinearGradientComputation,
    LayerGradientComputation,
    create_grad_feature_map,
)
from bmdal_reg.bmdal.selection import (
    Features,
    BatchTransform,
    PrecomputeTransform,
)


############# Acquisition Functions #############
def bmdal(
    method,
    base_kernel,
    kernel_transforms,
    model,
    batch_size,
    x_pool,
    x_train,
    y_train,
):
    """General wrapper for BMDAL selection methods"""
    # Get feature data
    device = next(model.parameters()).device
    y_train_torch = torch.tensor(y_train, dtype=torch.float32, device=device)
    train_data, pool_data = get_feature_data(x_train, x_pool, device)

    # Select batch with method and kernels
    with SuppressPrint():
        query_idx, _ = select_batch(
            batch_size=batch_size,
            models=[model],
            # Data
            data={"train": train_data, "pool": pool_data},
            y_train=y_train_torch,
            # Methods and Kernels
            base_kernel=base_kernel,
            kernel_transforms=kernel_transforms,
            selection_method=method,
        )

    # Return query idx and pool set
    query_idx = query_idx.cpu().numpy()
    return query_idx, x_pool[query_idx]


def random(model, x_pool, batch_size, x_train, y_train):
    """Random"""
    return bmdal(
        "random", "linear", [], model, batch_size, x_pool, x_train, y_train
    )


def bald(model, x_pool, batch_size, x_train, y_train):
    """MAXDIAG - BALD"""
    return bmdal(
        "maxdiag",
        "grad",
        [("sketch", [512]), ("acs-rf", [512, 1e-3, None])],
        model,
        batch_size,
        x_pool,
        x_train,
        y_train,
    )


def batch_bald(model, x_pool, batch_size, x_train, y_train):
    """MAXDET - BatchBALD"""
    return bmdal(
        "maxdet",
        "grad",
        [("sketch", [512]), ("train", [1e-3, None])],
        model,
        batch_size,
        x_pool,
        x_train,
        y_train,
    )


def badge(model, x_pool, batch_size, x_train, y_train):
    """Kmeans++ - Badge"""
    return bmdal(
        "kmeanspp",
        "grad",
        [("sketch", [512]), ("acs-rf", [512, 1e-3, None])],
        model,
        batch_size,
        x_pool,
        x_train,
        y_train,
    )


def bait(model, x_pool, batch_size, x_train, y_train):
    """Greedy Total Uncertainty Minimization - BAIT"""
    return bmdal(
        "bait",
        "grad",
        [("sketch", [512]), ("train", [1e-3, None])],
        model,
        batch_size,
        x_pool,
        x_train,
        y_train,
    )


def lcmd(model, x_pool, batch_size, x_train, y_train):
    """LCMD"""
    return bmdal(
        "lcmd",
        "grad",
        [("sketch", [512])],
        model,
        batch_size,
        x_pool,
        x_train,
        y_train,
    )


############# Posteriors Estimation #############
def get_posteriors(
    model,
    x_train,
    x_pool,
    sigma=1e-3,
    precomp_batch_size=32768,
    nn_batch_size=8192,
):
    """Compute the GP posterior variance"""
    # Base Kernel - Gradient
    grad_dict = {nn.Linear: LinearGradientComputation}
    grad_layers = []
    for layer in model.modules():
        if isinstance(layer, LayerGradientComputation):
            grad_layers.append(layer)
        elif type(layer) in grad_dict:
            grad_layers.append(grad_dict[type(layer)](layer))
    feature_map = create_grad_feature_map(
        model, grad_layers, use_float64=False
    )

    # Set up dataset
    device = next(model.parameters()).device
    train_data, pool_data = get_feature_data(x_train, x_pool, device)
    data = {"train": train_data, "pool": pool_data}
    features = {
        key: Features(feature_map, feature_data)
        for key, feature_data in data.items()
    }

    # Apply kernel transforms - Scale and Posterior
    apply_tfm(features, BatchTransform(batch_size=nn_batch_size))
    apply_tfm(features, PrecomputeTransform(batch_size=precomp_batch_size))
    apply_tfm(features, features["train"].scale_tfm(factor=None))
    apply_tfm(features, features["train"].posterior_tfm(sigma=sigma))
    apply_tfm(features, PrecomputeTransform(batch_size=precomp_batch_size))

    # Get posterior variance
    var = features["pool"].get_kernel_matrix_diag().cpu().numpy()
    var[var < 0] = 0  # avoid numerical error
    return var


def get_feature_data(x_train, x_pool, device):
    """Get feature data for BMDAL"""
    # BMDAL works with torch for parrallelization
    if isinstance(x_train, torch.Tensor):
        x_tr_torch = x_train.clone()
        x_pool_torch = x_pool.clone()
    else:
        x_tr_torch = torch.tensor(x_train, dtype=torch.float32, device=device)
        x_pool_torch = torch.tensor(x_pool, dtype=torch.float32, device=device)

    train_data = TensorFeatureData(x_tr_torch)
    pool_data = TensorFeatureData(x_pool_torch)
    return train_data, pool_data


def apply_tfm(features, tfm):
    """Apply kernel transforms"""
    for key in features:
        features[key] = tfm(features[key])


########## Helper Functions ##########
class SuppressPrint:
    """Context manager to suppress print statements inside a block."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")  # Redirect stdout to null
        return self  # Allows capturing return values if needed

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()  # Close the null file
        sys.stdout = self._original_stdout  # Restore stdout
