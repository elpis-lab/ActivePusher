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
                num_data = j + 1
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
    """Plot active learning results for all objects in 3x2 grid"""
    object_names = [
        "banana",
        "mug",
        "cracker_box_flipped",
        "mustard_bottle_flipped",
        "real_cracker_box_flipped",
        "real_mustard_bottle_flipped",
    ]
    for object_name in object_names:
        evaluate_physics(object_name, plot=False)

    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="deep")

    # Set font to Times New Roman and increase font sizes
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 20,  # Reduced from 26
            "axes.titlesize": 30,  # Increased to match title size
            "axes.labelsize": 30,  # Increased to match title size
            "xtick.labelsize": 35,  # Increased from 30 to 35
            "ytick.labelsize": 35,  # Increased from 30 to 35
            "legend.fontsize": 20,  # Reduced from 30
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

    # Create a 3x2 grid of subplots with square aspect ratio
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 24),
        sharey="row",  # Adjusted for square plots (16/2=8, 24/3=8)
    )
    axes = axes.flatten()  # Flatten to make indexing easier

    # Remove the space between subplots
    plt.subplots_adjust(wspace=0.8, hspace=0.8)  # Increased from 0.4 to 0.5

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
            # errorbar=("ci", 95),
            linewidth=5,  # Reduced from 7
            ax=ax,
            palette=custom_colors,  # Use our custom color mapping
            legend=False,  # Remove legend from all individual plots
        )

        # Set x-axis ticks - multiply by 10 for display
        x_ticks = np.arange(1, 11, 1)  # Keep original data range 1-10
        x_tick_labels = np.arange(10, 101, 10)  # Display as 10-100
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            x_tick_labels, rotation=0, fontsize=30
        )  # Increased from 30 to 35

        # Remove "flipped" from the title and format properly
        if object_name.startswith("real_"):
            # For real objects, format as "Object Name - Real"
            clean_name = (
                object_name.replace("real_", "")
                .replace("_flipped", "")
                .replace("_", " ")
                .title()
            )
            name = f"{clean_name} - Real"
        else:
            # For simulated objects, format as "Object Name - Sim"
            clean_name = (
                object_name.replace("_flipped", "").replace("_", " ").title()
            )
            name = f"{clean_name} - Sim"

        ax.set_title(name, fontsize=35)  # Increased from 24 to 30

        # Set y-axis limit and scale based on row (NO LOG SCALE)
        if i == 0 or i == 1:  # First row (banana and mug)
            ax.set_ylim(0.03, 0.17)
            # Set y-axis ticks for first row (starting from 0.03)
            y_ticks = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17]
        elif i == 2 or i == 3:  # Second row (cracker sim and mustard sim)
            ax.set_ylim(0.01, 0.16)
            # Set y-axis ticks for second row
            y_ticks = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
        else:  # Third row (cracker real and mustard real) - indices 4 and 5
            ax.set_ylim(0.01, 0.16)
            # Set y-axis ticks for third row
            y_ticks = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]

        # Force the limits and set ticks
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [f"{val:.2f}" for val in y_ticks], fontsize=30
        )  # Added fontsize=35

        # Remove any automatic scaling
        ax.autoscale(False)

        # Remove any ticks outside our range
        ax.tick_params(axis="y", which="minor", length=0)
        ax.tick_params(axis="y", which="major", length=5)

        # Set x-axis label
        ax.set_xlabel("")

        # Set custom y-axis label for the first plot only
        if i % 2 == 0:
            ax.set_ylabel("SE2 Error", fontsize=35)  # Reduced from 30
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

        # Remove right border for plots not in the rightmost column
        if i % 2 != 1:  # Not in rightmost column (1, 3, 5)
            ax.spines["right"].set_visible(False)

        # Remove left border for plots not in the leftmost column
        if i % 2 != 0:  # Not in leftmost column (0, 2, 4)
            ax.spines["left"].set_visible(False)

    # Add a common x-axis label for all plots
    fig.text(
        0.5, 0.08, "Number of Data", ha="center", fontsize=35
    )  # Moved down from 0.12 to 0.10

    # Create a single legend for the entire figure at the bottom
    # Get legend handles and labels from the first plot
    handles, labels = axes[0].get_legend_handles_labels()

    # If no handles were found, create them manually
    if not handles:
        from matplotlib.lines import Line2D

        handles = []
        labels = [
            "Pure Physics",
            "Residual Random",
            "MLP Random",
            "Residual Bait (Ours)",
            "MLP Bait",
        ]  # Reordered
        colors = [
            "#707b7c",
            "#2471a3",
            "#229954",
            "#a93226",
            "#d68910",
        ]  # Reordered colors to match
        for color in colors:
            handles.append(Line2D([0], [0], color=color, linewidth=5))

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),  # Moved up from 0.06 to 0.08
        frameon=True,
        framealpha=0.9,
        fontsize=35,  # Changed from 20 to 30 to match title size
        ncol=3,  # Changed to 1 column for vertical alignment
    )

    # Add some spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Keep the same bottom margin

    # Save the figure
    plt.savefig("results/learning/learning_all_objects.pdf", dpi=300)
    plt.savefig("results/learning/learning_all_objects.svg", dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()
