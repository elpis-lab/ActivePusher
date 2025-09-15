import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def main(obj_name, n_datas, types, reps_in_states=5):
    results = np.zeros((len(n_datas), len(types), reps_in_states, 6))
    for i, n_data in enumerate(n_datas):
        for j, ty in enumerate(types):
            model_type, learning_type, sampling = ty.split("_")
            name = (
                f"{obj_name}_{model_type}_{learning_type}_{n_data}_{sampling}"
            )
            res = np.load(f"results/planning/{name}_results.npy")
            results[i, j, :, :] = res

    # create a new plot two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for j, ty in enumerate(types):
        model_type, learning_type, sampling = ty.split("_")
        success = results[:, j, :, 0]
        success_mean = np.mean(success, axis=1)
        success_std = np.std(success, axis=1)
        error = results[:, j, :, 2]
        error_mean = np.mean(error, axis=1)
        error_std = np.std(error, axis=1)
        axs[0].plot(np.arange(len(success_mean)), success_mean, label=f"{ty}")
        axs[0].fill_between(
            np.arange(len(success_mean)),
            success_mean - success_std,
            success_mean + success_std,
            alpha=0.2,
        )
        axs[1].plot(np.arange(len(error_mean)), error_mean, label=f"{ty}")
        axs[1].fill_between(
            np.arange(len(error_mean)),
            error_mean - error_std,
            error_mean + error_std,
            alpha=0.2,
        )
    axs[0].legend()
    axs[1].legend()
    plt.show()
    return

    # Initialize storage
    success_rates = {
        f"{mt}{asamp}": [] for mt in model_types for asamp in active_samplings
    }
    tracking_errors = {
        f"{mt}{asamp}": [] for mt in model_types for asamp in active_samplings
    }

    # Data collection loop
    for rep in num:
        result_folder = f"results/planning_real/"

        for mt in model_types:
            for asamp in active_samplings:
                key = f"{mt}{asamp}"
                if len(success_rates[key]) < len(num_datas):
                    success_rates[key] = [[] for _ in num_datas]
                    tracking_errors[key] = [[] for _ in num_datas]

                for i, nd in enumerate(num_datas):
                    print(f"Processing batch {i+1} (data points: {nd})")
                    name = f"{obj_name}_{mt}_{exp_idx}_{nd}{asamp}"
                    folder = os.path.join(result_folder, name)
                    status_list = np.load(
                        os.path.join(folder, name + "_status_list.npy")
                    )
                    success_rate = np.sum(status_list) / len(status_list)
                    success_rates[key][i].append(success_rate)

                    states = np.load(
                        os.path.join(folder, name + "_states.npy"),
                        allow_pickle=True,
                    )
                    exec_states = np.load(
                        os.path.join(folder, name + "_exec_states.npy"),
                        allow_pickle=True,
                    )
                    # Get the tracking error for every state
                    tracking_error = get_tracking_mse(
                        states, exec_states, individual=True
                    )
                    tracking_errors[key][i].append(tracking_error)

                    # print the average number of steps
                    print(
                        name,
                        success_rate,
                        tracking_error,
                        np.mean([len(plan) for plan in states]),
                    )

    # Check if there's data for all 10 batches
    for key in success_rates:
        print(f"Model {key} has {len(success_rates[key])} data points")

    # Make results directory
    save_dir = "results/planning_real"
    os.makedirs(save_dir, exist_ok=True)

    # --- Plot with seaborn ---
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

    # Create nicer model names for the legend
    model_name_map = {
        "nn_random_regular": "MLP Random",
        "residual_random_regular": "Residual Random",
        "nn_bait_regular": "MLP Bait",
        "residual_bait_regular": "Residual Bait",
        "residual_bait_active": "Residual Bait (Active)",
    }

    # Alternative hex color options
    custom_colors = {
        "MLP Random": "#229954",  # Green
        "Residual Random": "#2471a3",  # Blue
        "MLP Bait": "#d68910",  # Yellow
        "Residual Bait": "#a93226",  # Red
        "Residual Bait (Active)": "#7d3c98",  # Purple
    }

    # Prepare DataFrame for success rates with batch numbers instead of data points
    records = []
    for key, per_nd in success_rates.items():
        for idx, rates in enumerate(per_nd):
            # Map to batch number (1-10) instead of data amount (20-200)
            batch_num = idx + 1  # Use 1-based indexing for batch numbers
            for r in rates:
                records.append(
                    {
                        "Batch Number": batch_num,
                        "Success Rate (%)": r * 100,
                        "Model": key,
                    }
                )
    df_success = pd.DataFrame.from_records(records)

    # Apply the mapping to create more readable labels
    df_success["Model Display"] = df_success["Model"].map(model_name_map)

    # Bar plot with manual positioning and custom colors
    plt.figure(figsize=(20, 12))

    # Group the data for plotting
    grouped = (
        df_success.groupby(["Batch Number", "Model Display"])[
            "Success Rate (%)"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Get unique models and batch numbers
    models = df_success["Model Display"].unique()
    batch_numbers = sorted(df_success["Batch Number"].unique())

    # Calculate bar width based on number of models
    n_models = len(models)
    width = 0.8 / n_models

    # Plot each model's bars manually with custom colors
    for i, model in enumerate(models):
        model_data = grouped[grouped["Model Display"] == model]
        # Calculate x positions - offset each model's bars
        x_pos = np.array(model_data["Batch Number"]) + width * (
            i - n_models / 2 + 0.5
        )

        # Get the color for this model from the custom palette
        color = custom_colors[model]

        # Plot the bars with error bars
        plt.bar(
            x_pos,
            model_data["mean"],
            width=width * 0.9,
            yerr=model_data["std"],
            label=model,
            color=color,
            capsize=3,
        )

    # Set up the axes
    x_ticks = range(1, 11)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=30)
    ax.set_xlim(0.5, 5.5)  # Ensure we can see all 10 batches

    # Set y-axis configurations
    y_ticks = range(0, 101, 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=30)
    ax.set_ylim(0, 100)

    # Labels and grid
    ax.set_xlabel("Number of Batches", fontsize=30)
    ax.set_ylabel("Success Rate (%)", fontsize=30)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Legend
    plt.legend(
        frameon=True,
        framealpha=0.9,
        fontsize=26,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
    )

    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Save the figure
    plt.savefig(
        os.path.join(save_dir, "planning_success_rate_barplot_manual.pdf"),
        dpi=300,
    )
    plt.savefig(
        os.path.join(save_dir, "planning_success_rate_barplot_manual.svg"),
        dpi=300,
    )

    # Explicitly verify the DataFrame includes batch 10
    batch_counts = df_success["Batch Number"].value_counts().sort_index()
    print("Batches in DataFrame:", batch_counts)

    # Prepare DataFrame for tracking errors with batch numbers
    records = []
    for key, per_nd in tracking_errors.items():
        for idx, errs in enumerate(per_nd):
            # Map to batch number (1-10) instead of data amount (20-200)
            batch_num = idx + 1
            for e in errs:
                records.append(
                    {
                        "Batch Number": batch_num,  # Use batch number
                        "Tracking RMSE": np.sqrt(e),
                        "Model": key,
                    }
                )
    df_track = pd.DataFrame.from_records(records)

    # Apply the mapping to create more readable labels
    df_track["Model Display"] = df_track["Model"].map(model_name_map)

    # Line plot with consistent legend style and custom colors
    plt.figure(figsize=(16, 12))
    sns.lineplot(
        data=df_track,
        x="Batch Number",
        y="Tracking RMSE",
        hue="Model Display",
        palette=custom_colors,
        marker="o",
        err_style="band",
        errorbar="sd",
        linewidth=7,
    )

    # Get the current axis
    ax = plt.gca()

    # Set x-axis ticks to values 1 through 5
    x_ticks = range(1, 11)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=30)

    # Set x-axis limits to match the bar plot
    ax.set_xlim(0.5, 5.5)

    # Set specific y-axis ticks and labels
    y_ticks = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{val:.2f}" for val in y_ticks], fontsize=30)

    # Set y-axis limits
    ax.set_ylim(0.0, 0.45)

    plt.xlabel("Number of Batches", fontsize=30)
    plt.ylabel("Tracking Error (RMSE)", fontsize=30)

    # Match the legend style from the bar plot
    plt.legend(
        frameon=True,
        framealpha=0.9,
        fontsize=26,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend at bottom
    plt.savefig(os.path.join(save_dir, "planning_tracking_error_seaborn.pdf"))
    plt.savefig(
        os.path.join(save_dir, "planning_tracking_error_seaborn.svg"), dpi=300
    )
    plt.show()

    # Save raw data
    with open(os.path.join(save_dir, "success_rates.pkl"), "wb") as f:
        pickle.dump(success_rates, f)
    with open(os.path.join(save_dir, "tracking_errors.pkl"), "wb") as f:
        pickle.dump(tracking_errors, f)

    print(f"Data and plots saved to '{save_dir}'")


if __name__ == "__main__":
    obj_names = [
        "mustard_bottle_flipped",
        "cracker_box_flipped",
    ]
    real_obj_names = [
        "real_mustard_bottle_flipped",
        "real_cracker_box_flipped",
    ]

    n_datas = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    real_n_datas = [20, 40, 60, 80, 100]
    types = [
        "mlp_random_regular",
        # "mlp_random_active",
        "residual_bait_regular",
        "residual_bait_active",
    ]

    for obj_name in obj_names:
        main(obj_name, n_datas, types, reps_in_states=5)
    for obj_name in real_obj_names:
        main(obj_name, real_n_datas, types, reps_in_states=1)
