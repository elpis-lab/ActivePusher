import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def load_closed_loop_results(obj_names, n_datas, types, reps_in_states=5):
    """Load closed-loop planning results for all objects and methods"""
    all_data = []

    # Load our method results
    results = np.zeros(
        (len(obj_names), len(n_datas), len(types), reps_in_states, 6)
    )
    for i, obj_name in enumerate(obj_names):
        for j, n_data in enumerate(n_datas):
            for k, ty in enumerate(types):
                model_type, learning_type, sampling = ty.split("_")
                name = f"{obj_name}_{model_type}_{learning_type}_{n_data}_{sampling}"
                res = np.load(
                    f"results/planning/{name}_results_closed_loop.npy"
                )
                results[i, j, k, :, :] = res

    # Convert to DataFrame format
    for i, obj_name in enumerate(obj_names):
        for k, ty in enumerate(types):
            success_rates = results[
                i, :, k, :, 0
            ]  # Shape: (n_batches, n_runs)

            # Create individual data points for each run and batch
            for batch_idx in range(success_rates.shape[0]):
                for run_idx in range(success_rates.shape[1]):
                    all_data.append(
                        {
                            "Data Size": n_datas[batch_idx],
                            "Success Rate": success_rates[batch_idx, run_idx],
                            "Method": ty,
                            "Object": obj_name,
                            "Run": run_idx,
                        }
                    )

    # Load HACMAN results
    for i, obj_name in enumerate(obj_names):
        hacman_results = np.load(f"results/hacman/{obj_name}_results.npy")
        x_hacman = np.arange(1000, 100001, 1000)

        # Add zero value at x=10 for HACMAN
        x_hacman_with_zero = np.concatenate([[10], x_hacman])
        hacman_results_with_zero = np.concatenate([[0], hacman_results])

        for j, (data_size, success_rate) in enumerate(
            zip(x_hacman_with_zero, hacman_results_with_zero)
        ):
            all_data.append(
                {
                    "Data Size": data_size,
                    "Success Rate": success_rate,
                    "Method": "hacman",
                    "Object": obj_name,
                    "Run": 0,  # Single run for HACMAN
                }
            )

    return pd.DataFrame(all_data)


def main():
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="deep")

    # Set font to Times New Roman and increase font sizes
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 20,
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 35,
            "ytick.labelsize": 35,
            "legend.fontsize": 30,
        }
    )

    # Create nicer model names for the legend
    model_name_map = {
        "mlp_random_regular": "Regular Planning with MLP Random",
        "residual_bait_active": "Active Planning with Residual Bait (Ours)",
        "hacman": "HACMan",
    }

    # Custom colors matching the other scripts
    custom_colors = {
        "Regular Planning with MLP Random": "#229954",  # Green
        "Active Planning with Residual Bait (Ours)": "#7d3c98",  # Purple
        "HACMan": "#ff9933",  # Orange
    }

    # Object names and parameters
    obj_names = [
        "mustard_bottle_flipped",
        "cracker_box_flipped",
    ]

    n_datas = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    types = [
        "mlp_random_regular",
        "residual_bait_active",
    ]

    # Load results
    df = load_closed_loop_results(obj_names, n_datas, types, reps_in_states=5)

    # Create single plot with extended height for legend
    fig, ax = plt.subplots(figsize=(16, 10))  # Extended height from 8 to 12

    # Map method names for display
    df["Method"] = df["Method"].map(model_name_map)

    # Create separate plots for each object to control markers
    for obj_idx, obj_name in enumerate(obj_names):
        obj_df = df[df["Object"] == obj_name].copy()

        # Choose marker and line width based on object
        marker = (
            "o" if obj_name == "mustard_bottle_flipped" else "s"
        )  # circles for mustard, squares for cracker
        linewidth = (
            8 if obj_name == "mustard_bottle_flipped" else 3
        )  # thinner for mustard, thicker for cracker
        markersize = (
            12 if obj_name == "mustard_bottle_flipped" else 9
        )  # smaller for mustard, bigger for cracker

        # Plot success rates using seaborn
        sns.lineplot(
            data=obj_df,
            x="Data Size",
            y="Success Rate",
            hue="Method",
            marker=marker,
            err_style=None,  # Remove error bands
            linewidth=linewidth,  # Different line widths for different objects
            markersize=markersize,  # Different marker sizes for different objects
            linestyle="-",  # All lines use solid style
            ax=ax,
            palette=custom_colors,
            legend=False,
        )

    # Set log scale for x-axis
    ax.set_xscale("log", base=10)

    # Removed title - no title needed

    # Set x-axis label with lower position
    ax.set_xlabel(
        "Data Size (Log scale)", fontsize=35, labelpad=12
    )  # Added labelpad to move label lower

    # Set y-axis label
    ax.set_ylabel("Success Rate (Closed-loop)", fontsize=35)

    # Set y-axis limits
    ax.set_ylim(-0.02, 1.02)

    # Enable grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set custom x-axis scale to compress the gap between 100 and 1000
    # Create custom tick positions and labels
    x_ticks = [10, 100, 1000, 10000, 100000]
    x_tick_labels = [
        "1e1",
        "1e2",
        "1e3",
        "1e4",
        "1e5",
    ]

    # Set the ticks and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=30)

    # Set custom x-axis limits to compress the gap
    ax.set_xlim(9, 110000)  # This will compress the space between 100 and 1000

    # Set y-axis ticks
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        [f"{val:.1f}" for val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]], fontsize=30
    )

    # Create legend at the bottom outside the figure
    from matplotlib.lines import Line2D

    # Create legend handles manually for methods
    method_handles = []
    method_labels = []
    for method in [
        "Active Planning with Residual Bait (Ours)",
        "Regular Planning with MLP Random",
        "HACMan",
    ]:
        color = custom_colors[method]
        method_handles.append(
            Line2D(
                [0], [0], color=color, marker="o", linewidth=5, markersize=10
            )
        )
        method_labels.append(method)

    # Create legend handles for objects (different markers and line styles)
    object_handles = []
    object_labels = []
    markers = ["o", "s"]  # circles and squares
    line_styles = ["-", "-"]  # both solid
    line_widths = [6, 2]  # thicker for mustard, thinner for cracker
    marker_sizes = [12, 8]  # bigger for mustard, smaller for cracker
    object_names = ["Push to Region", "Push to Edge"]  # New names
    for i, obj_name in enumerate(obj_names):
        object_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=markers[i],
                linewidth=line_widths[i],  # Match the actual line widths
                markersize=marker_sizes[i],  # Match the actual marker sizes
                linestyle=line_styles[i],
            )
        )
        object_labels.append(object_names[i])  # Use the new names

    # Combine all handles and labels
    all_handles = method_handles + object_handles
    all_labels = method_labels + object_labels

    # Create legend at the bottom - moved lower
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.04,
        ),  # Moved even lower with more negative y value
        frameon=True,
        framealpha=0.9,
        fontsize=35,
        ncol=2,  # Arrange in 2 columns
    )

    # Adjust layout to make room for legend - increased bottom margin
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.42
    )  # Increased from 0.25 to 0.35 to accommodate lower legend

    # Save the figure with bbox_inches='tight' to ensure legend is included
    save_dir = "results/planning"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        f"{save_dir}/rl_planning_results.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{save_dir}/rl_planning_results.svg", dpi=300, bbox_inches="tight"
    )
    # plt.show()

    print(
        f"Closed-loop planning plot saved to {save_dir}/rl_planning_results.pdf"
    )


if __name__ == "__main__":
    main()
