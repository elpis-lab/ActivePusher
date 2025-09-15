import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def main(obj_names, n_datas, types, reps_in_states=5):
    # Plot results
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
    x_results = np.array(n_datas)

    # create a new plot two subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, obj_name in enumerate(obj_names):
        for k, ty in enumerate(types):
            success = results[i, :, k, :, 0]  # (len(n_datas), reps_in_states)
            success_mean = success.mean(axis=1)
            success_std = success.std(axis=1)

            ax.plot(
                x_results,
                success_mean,
                marker="o",
                label=f"{obj_name}_{ty}",
            )
            ax.fill_between(
                x_results,
                success_mean - success_std,
                success_mean + success_std,
                alpha=0.2,
            )

    # Plot HACMAN
    for i, obj_name in enumerate(obj_names):
        hacman_results = np.load(f"results/hacman/{obj_name}_results.npy")
        x_hacman = np.arange(1000, 100001, 1000)

        ax.plot(
            x_hacman,
            hacman_results,
            linestyle="--",
            linewidth=2,
            label=f"{obj_name}_HACMAN",
        )

    # Log-scale x so 10–100 and 1e3–1e5 can live together
    ax.set_xscale("log", base=10)

    # Cosmetics
    xmin = min(x_results.min(), x_hacman.min())
    xmax = max(x_results.max(), x_hacman.max())
    ax.set_xlim(xmin * 0.9, xmax * 1.1)

    ax.set_xlabel("Data size / Training steps (log scale)")
    ax.set_ylabel("Success rate")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend()
    fig.suptitle(f"Closed-loop planning results")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    obj_names = [
        "mustard_bottle_flipped",
        "cracker_box_flipped",
    ]

    n_datas = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    types = [
        "mlp_random_regular",
        "residual_bait_active",
    ]

    main(obj_names, n_datas, types, reps_in_states=5)
