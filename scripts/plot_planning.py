import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def load_results_with_runs(
    obj_name, n_datas, types, model_name_map, num_runs=5
):
    """Load results for a single object"""
    all_data = []
    print(f"\n=== Loading data for {obj_name} ===")

    # Load the original data once
    results = np.zeros((len(n_datas), len(types), num_runs, 6))

    for i, n_data in enumerate(n_datas):
        for j, ty in enumerate(types):
            model_type, learning_type, sampling = ty.split("_")
            name = (
                f"{obj_name}_{model_type}_{learning_type}_{n_data}_{sampling}"
            )
            res = np.load(f"results/planning/{name}_results.npy")
            results[i, j, :, :] = res
            print(f"  Loaded: {name}")

    # Convert to DataFrame format - create individual data points for each run
    for i, ty in zip(range(len(types)), types):
        model_type, learning_type, sampling = ty.split("_")
        success_rates = results[:, i, :, 0]  # Shape: (n_batches, n_runs)
        errors = results[:, i, :, 2]  # Shape: (n_batches, n_runs)
        print(f"Success rates: {success_rates.shape}")

        model_display = model_name_map[ty]

        # Create individual data points for each run and batch
        for batch_idx in range(success_rates.shape[0]):
            for run_idx in range(success_rates.shape[1]):
                all_data.append(
                    {
                        "Batch Number": batch_idx + 1,
                        "Success Rate": success_rates[batch_idx, run_idx],
                        "Tracking Error": errors[batch_idx, run_idx],
                        "Method": model_display,
                        "Run": run_idx,
                    }
                )

    df = pd.DataFrame(all_data)
    print(f"  Total data points loaded: {len(df)}")
    print(f"  Unique runs: {df['Run'].nunique()}")
    print(f"  Unique methods: {df['Method'].unique()}")
    print(f"  Unique batch numbers: {sorted(df['Batch Number'].unique())}")

    return df


def main():
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="deep")

    # Set font to Times New Roman and increase font sizes
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 20,
            "axes.titlesize": 35,
            "axes.labelsize": 30,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 18,
        }
    )

    # Create nicer model names for the legend
    model_name_map = {
        "mlp_random_regular": "Regular Planning with MLP Random",
        "residual_random_regular": "Regular Planning with Residual Random",
        "mlp_bait_regular": "Regular Planning with MLP Bait",
        "residual_bait_regular": "Regular Planning with Residual Bait",
        "residual_bait_active": "Active Planning with Residual Bait (Ours)",
    }

    # Alternative hex color options
    custom_colors = {
        "Regular Planning with MLP Random": "#229954",  # Green
        "Regular Planning with Residual Random": "#2471a3",  # Blue
        "Regular Planning with MLP Bait": "#d68910",  # Yellow
        "Regular Planning with Residual Bait": "#a93226",  # Red
        "Active Planning with Residual Bait (Ours)": "#7d3c98",  # Purple
    }

    # Object names and parameters
    obj_names = [
        "cracker_box_flipped",
        "mustard_bottle_flipped",
        "real_cracker_box_flipped",
        "real_mustard_bottle_flipped",
    ]

    # Different data specifications for simulated vs real objects
    sim_n_datas = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    real_n_datas = [20, 40, 60, 80, 100]

    types = [
        "mlp_random_regular",
        "residual_bait_regular",
        "residual_bait_active",
    ]

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    # Plot each object
    for obj_idx, obj_name in enumerate(obj_names):
        ax1 = axes[obj_idx]

        # Choose data specifications based on object type
        if obj_name.startswith("real_"):
            n_datas = real_n_datas
            num_runs = 1
            print(
                f"Using real object specifications: {n_datas} data points, {num_runs} runs"
            )
        else:
            n_datas = sim_n_datas
            num_runs = 5
            print(
                f"Using simulated object specifications: {n_datas} data points, {num_runs} runs"
            )

        # Load results with appropriate specifications
        df = load_results_with_runs(
            obj_name, n_datas, types, model_name_map, num_runs
        )

        # Debug: Check if we have multiple runs for standard deviation
        unique_runs = df["Run"].nunique()
        print(f"\nPlotting {obj_name}: {unique_runs} unique runs")

        if unique_runs == 1:
            print(
                f"  WARNING: Only 1 run available for {obj_name} - no standard deviation bands will be shown!"
            )

        # Plot success rates using seaborn
        sns.lineplot(
            data=df,
            x="Batch Number",
            y="Success Rate",
            hue="Method",
            marker="o",
            err_style="band",
            errorbar="sd",
            linewidth=4,
            markersize=8,
            ax=ax1,
            palette=custom_colors,
            legend=False,
        )

        # Create second y-axis for tracking errors
        ax2 = ax1.twinx()

        # Plot tracking errors using seaborn
        sns.lineplot(
            data=df,
            x="Batch Number",
            y="Tracking Error",
            hue="Method",
            marker="s",
            err_style="band",
            errorbar="sd",
            linewidth=4,
            markersize=8,
            linestyle="--",
            ax=ax2,
            palette=custom_colors,
            legend=False,
        )

        # Remove x-axis label from individual plots
        ax1.set_xlabel("", fontsize=20)

        # Only show y-axis labels on the leftmost subplots
        if obj_idx % 2 == 0:  # Left column (0, 2)
            ax1.set_ylabel("Success Rate", fontsize=35, color="black")
            ax1.tick_params(axis="y", labelcolor="black", labelsize=30)
        else:
            ax1.set_ylabel("", fontsize=35, color="black")
            ax1.tick_params(
                axis="y", labelcolor="black", labelsize=30, labelleft=False
            )

        # Clean object name for title
        clean_name = (
            obj_name.replace("_", " ").replace("flipped", "").title().strip()
        )
        if "cracker" in clean_name.lower():
            clean_name = "Push to Edge"
        if "mustard" in clean_name.lower():
            clean_name = "Push to Region"
        if obj_name.startswith("real_"):
            clean_name = clean_name.replace("Real ", "") + " - Real"
        else:
            clean_name = clean_name + " - Sim"

        ax1.set_title(clean_name, fontsize=35)

        # Set x-axis ticks based on object type
        if obj_name.startswith("real_"):
            ax1.set_xticks(range(1, 6))  # 5 data points for real objects
            ax1.set_xticklabels(
                [str(x) for x in range(20, 101, 20)], fontsize=24
            )
            ax1.set_xlim(0.9, 5.1)  # Bottom row (real results)
        else:
            ax1.set_xticks(
                range(1, 11)
            )  # 10 data points for simulated objects
            ax1.set_xticklabels(
                [str(x) for x in range(10, 101, 10)], fontsize=24
            )
            ax1.set_xlim(0.8, 10.2)  # Top row (simulation)

        ax1.set_ylim(0, 1)
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Configure right y-axis (Tracking Error)
        # Only show y-axis labels on the rightmost subplots
        if obj_idx % 2 == 1:  # Right column (1, 3)
            ax2.set_ylabel("SE2 Error", fontsize=35, color="black")
            ax2.tick_params(axis="y", labelcolor="black", labelsize=30)
        else:
            ax2.set_ylabel("", fontsize=35, color="black")
            ax2.tick_params(
                axis="y", labelcolor="black", labelsize=30, labelright=False
            )
        ax2.set_ylim(0, 0.2)

        # Set manual y-axis tick labels for tracking error
        y_ticks = [0.0, 0.04, 0.08, 0.12, 0.16, 0.2]
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels([f"{val:.2f}" for val in y_ticks], fontsize=30)

        # Create combined legend only for the first subplot
        if obj_idx == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Don't create legend here, we'll do it at the bottom

    # Add a common x-axis label for all plots at the bottom center
    fig.text(
        0.5, 0.16, "Number of Data", ha="center", fontsize=35
    )  # Changed from 24 to 35

    # Create a single legend for the entire figure at the bottom
    # Create legend handles manually since seaborn plots have legend=False
    from matplotlib.lines import Line2D

    # Create handles for methods (by color)
    method_handles = []
    method_labels = []
    for i, ty in enumerate(types):
        model_display = model_name_map[ty]
        color = custom_colors[model_display]
        method_handles.append(
            Line2D(
                [0], [0], color=color, marker="o", linewidth=5, markersize=10
            )
        )
        method_labels.append(model_display)

    # Create handles for line styles (success rate vs tracking error)
    style_handles = []
    style_labels = []
    style_handles.append(
        Line2D([0], [0], color="black", linewidth=5, linestyle="-")
    )
    style_labels.append("Success Rate")
    style_handles.append(
        Line2D([0], [0], color="black", linewidth=5, linestyle="--")
    )
    style_labels.append("Tracking Error")

    # Combine all handles and labels
    all_handles = method_handles + style_handles
    all_labels = method_labels + style_labels

    # Create legend at the bottom
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),  # Moved up from -0.2 to 0.05
        frameon=True,
        framealpha=0.9,
        fontsize=35,  # Changed from 18 to 35
        ncol=2,  # Arrange in 2 columns
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.22
    )  # Increased from 0.12 to 0.2 to give more space for legend

    # Save the figure
    save_dir = "results/planning"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}/active_planning_results.pdf", dpi=300)
    plt.savefig(f"{save_dir}/active_planning_results.svg", dpi=300)
    # plt.show()

    print(f"Combined plot saved to {save_dir}/active_planning_results.pdf")


if __name__ == "__main__":
    main()
