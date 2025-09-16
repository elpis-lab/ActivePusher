import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from active_learning.kernel import get_posteriors

# from lie_group.lie_se2 import se2_stats
from utils import DataLoader, parse_args, set_seed, get_names
from geometry.pose import angle_diff
from geometry.object_model import get_obj_shape
from models.physics import push_physics
from models.torch_model import MLP, MLPVar, MLPEvidential
from models.torch_model import Physics, ResidualPhysics
from models.torch_model import ResidualPhysicsVar, ResidualPhysicsEvidential
from models.torch_loss_se2 import (
    se2_split_loss,
    mse_se2_loss,
    nll_se2_loss,
)
from models.torch_loss_se2 import beta_nll_se2_loss
from models.torch_loss_se2 import evidential_se2_loss
from models.model import TorchModel


def load_model(
    model_type="mlp",
    equation=None,  # physics equation
    use_var=0,  # 0: no variance, 1: variance, 2: evidential
    in_dim=3,
    out_dim=3,
    hidden=32,
    dropout=0,
    lr=1e-3,
    batch_size=16,
    epochs=1000,
    device=None,
):
    """
    Load model wrapper function
    Return a model with Sklearn style API for PyTorch
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Select which model class to use
    require_training = True
    if model_type == "mlp":
        if not use_var:
            model_class = lambda: MLP(in_dim, out_dim, hidden, dropout)
        elif use_var == 1:
            model_class = lambda: MLPVar(in_dim, out_dim, hidden, dropout)
        elif use_var == 2:
            model_class = lambda: MLPEvidential(
                in_dim, out_dim, hidden, dropout
            )
    elif model_type == "residual":
        if not use_var:
            model_class = lambda: ResidualPhysics(
                in_dim, out_dim, equation, hidden, dropout
            )
        elif use_var == 1:
            model_class = lambda: ResidualPhysicsVar(
                in_dim, out_dim, equation, hidden, dropout
            )
        elif use_var == 2:
            model_class = lambda: ResidualPhysicsEvidential(
                in_dim, out_dim, equation, hidden, dropout
            )
    elif model_type == "physics":
        model_class = lambda: Physics(equation)
        require_training = False
    else:
        raise ValueError(f"Model type {model_type} not supported")

    # Use NLL loss if variance is predicted
    if use_var == 0:
        loss_fn = se2_split_loss  # mse_se2_loss
    elif use_var == 1:
        loss_fn = nll_se2_loss  # beta_nll_se2_loss
    elif use_var == 2:
        loss_fn = evidential_se2_loss
    score_fn = se2_split_loss  # mse_se2_loss

    # Get a wrapper for the model
    model = TorchModel(
        model_class,
        optimizer=torch.optim.Adam,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        loss_fn=loss_fn,
        score_fn=score_fn,
        verbose=1,
        require_training=require_training,
        device=device,
    )
    return model


def get_push_physics(model_type, obj_size):
    """
    Get the push physics function with given object size
    if model_type requires physics.
    Return None otherwise.
    """

    def push_physics_with_size(param):
        """Push physics function with given object size."""
        return push_physics(param, obj_size[:2], relative=True)

    if model_type == "residual" or model_type == "physics":
        return push_physics_with_size
    else:
        return None


def evaluate_results(pred, true, verbose=False):
    """Evaluate results"""
    pred = np.asarray(pred)
    true = np.asarray(true)
    # SE2 error
    pos_error = np.mean(np.linalg.norm(true[:, :2] - pred[:, :2], axis=1))
    rot_error = np.mean(np.abs(angle_diff(true[:, 2], pred[:, 2])))
    se2_loss = pos_error + 0.2 * rot_error
    if verbose:
        print(
            f"SE2 Error: {se2_loss}"
            + f" - Position Error: {pos_error}"
            + f" - Rotation Error: {rot_error}"
        )

    return se2_loss, pos_error, rot_error


def test(obj_name, model_type, use_var=0, n_data=200, plot=True):
    """Train a model."""
    name = f"{obj_name}_{model_type}_{use_var}_{n_data}"

    # Load data
    model_name, data_name = get_names(obj_name)
    data_loader = DataLoader(data_name, val_size=1000, test_size=1000)
    dataset = data_loader.load_data()

    # Load model
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    physics_eq = get_push_physics(model_type, obj_shape[:2])
    model = load_model(model_type, physics_eq, use_var, epochs=1000)

    # Test training
    tr_losses, val_losses = model.fit(
        dataset["x_pool"][:n_data],
        dataset["y_pool"][:n_data],
        dataset["x_val"],
        dataset["y_val"],
    )
    if plot:
        plt.plot(np.arange(len(tr_losses)), tr_losses, label="Training")
        plt.plot(np.arange(len(val_losses)), val_losses, label="Validation")
        plt.legend()
        plt.show()
    # model.save(f"test_{name}.pth")
    # model.load(f"test_{name}.pth")

    # Test active learning model
    # name = f"{obj_name}_residual_bait_0_100"
    # model.load(f"results/models/{name}.pth")
    # used_indices = pickle.load(
    #     open(f"results/learning/idx_used_{obj_name}_{model_type}.pkl", "rb")
    # )
    # used_indices = used_indices["bait"][0][:n_data]
    # data_loader = DataLoader(data_name, val_size=1000)
    # dataset = data_loader.load_data()
    # x_train = dataset["x_pool"][used_indices]

    # Test the active sampling method
    x_train = dataset["x_pool"][:n_data]
    pred = model.predict(dataset["x_val"])
    var = get_posteriors(
        model.model,
        x_train,
        dataset["x_val"],
        sigma=5e-3,
    )
    # See if the prediction on lower-variance points is better
    n_top = int(len(var) * 0.2)
    top_idx = np.argsort(var)[:n_top]
    bottom_idx = np.argsort(var)[n_top:]
    print("Overall results:")
    res = evaluate_results(pred, dataset["y_val"], plot)
    print("Top results:")
    res = evaluate_results(pred[top_idx], dataset["y_val"][top_idx], plot)
    print("Bottom results:")
    res = evaluate_results(
        pred[bottom_idx], dataset["y_val"][bottom_idx], plot
    )
    return res


if __name__ == "__main__":
    args = parse_args(
        [
            ("obj_name", "mustard_bottle_flipped"),
            ("model_type", "residual"),
            ("use_var", 0, float),
            ("n_data", 200, int),
        ]
    )
    set_seed(42)

    test(args.obj_name, args.model_type, args.use_var, args.n_data)
