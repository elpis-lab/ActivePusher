import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch

from models.physics import push_physics
from geometry.object_model import get_obj_shape
from models.torch_loss_se2 import se2_split_loss
from utils import DataLoader, get_names


def evaluate_physics(object_name, plot=False):
    # Load data
    model_name, data_name = get_names(object_name)
    data_loader = DataLoader(data_name, test_size=1000)
    datasets = data_loader.load_data()
    x_test = torch.from_numpy(datasets["x_test"])
    y_test = torch.from_numpy(datasets["y_test"])
    # Load object
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")

    # Run physics
    y_pred = push_physics(x_test, obj_size=obj_shape[:2])
    # Calculate SE2
    error = se2_split_loss(y_pred, y_test).cpu().numpy()
    pos_error = torch.norm(y_pred[:, :2] - y_test[:, :2], dim=1)
    rot_error = torch.abs(
        (y_test[:, 2] - y_pred[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
    )
    # Save the data
    pickle.dump(
        error, open(f"results/learning/loss_{object_name}_physics.pkl", "wb")
    )
    if plot:
        print(f"PhysicsError: {error.mean()}")
        print(f"Physics Position error: {pos_error.mean()}")
        print(f"Physics Rotation error: {rot_error.mean()}")


def create_plot_data(object_name):
    """Load data and create dataframe for an object"""
    # Load active training files
    # Load result - shape(n_experiments, n_loop)
    res = pickle.load(
        open("results/learning/loss_" + object_name + "_residual.pkl", "rb")
    )
    mlp = pickle.load(
        open("results/learning/loss_" + object_name + "_mlp.pkl", "rb")
    )
    res_bait = res["bait"]
    res_random = res["random"]
    mlp_bait = mlp["bait"]
    mlp_random = mlp["random"]

    physics = pickle.load(
        open("results/learning/loss_" + object_name + "_physics.pkl", "rb")
    )
    # extend it to be the same shape
    physics = physics * np.ones_like(res_bait)

    scores = [physics, mlp_random, res_random, mlp_bait, res_bait]
    keys = [
        "Pure Physics",
        "MLP Random",
        "Residual Random",
        "MLP Bait",
        "Residual Bait",
    ]

    # Convert to a format suitable for seaborn
    data = []
    for i, score in enumerate(scores):
        for exp_idx in range(score.shape[0]):  # For each experiment
            for j, val in enumerate(score[exp_idx]):
                num_data = (j + 1) * 10
                title = object_name.replace("_", " ").title()
                data.append(
                    {
                        "Method": keys[i],
                        "Number of Data": num_data,
                        "SE2 Error": val,
                        "Object": title,
                    }
                )

    # Create a DataFrame
    return pd.DataFrame(data)


def main():
    """Plot active learning results for all objects horizontally"""
    object_names = [
        # "real_cracker_box_flipped",
        # "real_mustard_bottle_flipped",
        "cracker_box_flipped",
        "mustard_bottle_flipped",
        "banana",
        "mug",
    ]
    for object_name in object_names:
        evaluate_physics(object_name, plot=False)

    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="deep")

    # Set font to Times New Roman and increase font sizes
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 26,
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 30,
        }
    )

    # Define custom colors for each method (same as in plot_planning.py)
    custom_colors = {
        "Pure Physics": "#707b7c",  # Grey
        "MLP Random": "#229954",  # Green
        "Residual Random": "#2471a3",  # Blue
        "MLP Bait": "#d68910",  # Yellow
        "Residual Bait": "#a93226",  # Red
    }

    # Create a single figure with subplots horizontally
    fig, axes = plt.subplots(
        1, len(object_names), figsize=(45, 15), sharey=True
    )

    # Remove the space between subplots
    plt.subplots_adjust(wspace=0)

    # Create the formatter for y-axis
    formatter = ScalarFormatter(useOffset=False, useMathText=False)

    # Generate plots for each object
    for i, object_name in enumerate(object_names):
        # Load data for this object
        df = create_plot_data(object_name)

        # Plot on current subplot
        ax = axes[i]
        sns.lineplot(
            data=df,
            x="Number of Data",
            y="SE2 Error",
            hue="Method",
            marker="o",
            err_style="band",  # Confidence interval as a shaded band
            errorbar=("sd", 1),  # ~1 standard deviation
            linewidth=7,  # Increase line thickness
            ax=ax,
            palette=custom_colors,  # Use our custom color mapping
            legend=False if i > 0 else True,  # Only show legend on first plot
        )

        # Set specific y-axis ticks and labels (only for the first plot)
        # y_ticks = np.arange(0.02, 0.17, 0.02)
        # ax.set_yticks(y_ticks)
        # if i == 0:
        #     ax.set_yticklabels([f"{val:.1f}" for val in y_ticks])
        # else:
        #     ax.set_yticklabels([])  # No labels for other plots

        # Set y-axis limits to ensure all ticks are visible
        ax.set_ylim(0.0, 0.17)

        # # Apply formatter to y-axis (to avoid scientific notation)
        # formatter.set_scientific(False)
        # ax.yaxis.set_major_formatter(formatter)

        # Set x-axis ticks
        x_ticks = np.arange(1, 11, 1) * 10
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45)

        # Comment out or remove the title setting
        name = object_name.replace("_", " ").title()
        ax.set_title(name, fontsize=25)

        # Set x-axis label
        ax.set_xlabel("")

        # Set custom y-axis label for the first plot only
        if i == 0:
            ax.set_ylabel("SE2 Error", fontsize=25)
        else:
            ax.set_ylabel("")

        # Enable grid on all plots with same style
        ax.grid(
            True,
            which="both",
            axis="both",
            linestyle="-",
            color="lightgray",
            alpha=0.7,
        )

        # Remove right border for all plots except the last one
        if i < len(object_names) - 1:
            ax.spines["right"].set_visible(False)

        # Remove left border for all plots except the first one
        if i > 0:
            ax.spines["left"].set_visible(False)

    # Add a common x-axis label for all plots
    fig.text(0.5, 0.15, "Number of Training Data", ha="center", fontsize=25)
    # Create a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.15),
        frameon=True,
        framealpha=0.9,
        fontsize=26,
        ncol=5,
    )

    # Add some spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend at the bottom

    # Save the figure
    plt.savefig("results/learning/all_objects.pdf", dpi=300)
    plt.savefig("results/learning/all_objects.svg", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
