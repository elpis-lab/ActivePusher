import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from utils import DataLoader, parse_args, get_names, set_seed
from geometry.object_model import get_obj_shape
from run_plans import run_plans
from planning_edge import run_planning as run_planning_edge
from planning_region import run_planning as run_planning_region
from planning.ompl_utils import set_ompl_seed
from run_plans_pool import run_plans_pool
from run_plans_sim import run_plans_sim


def run_plans_closed_loop(
    evaluate_fn,
    model_type,
    learning_type,
    n_data,
    active_sampling,
    plan_states,
    plan_controls,
    obj_name,
    env,
    reps_in_states=1,
    max_retries=1,
    replan_time=1,
    verbose=False,
):
    """Run the plans in closed loop manner"""
    # Load data and model
    model_name, data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    data_loader = DataLoader(data_name)
    datasets = data_loader.load_data()

    # First run the plans once
    res, success, invalid, exec_states = run_plans(
        evaluate_fn, plan_states, plan_controls, obj_name, env, 1, verbose
    )

    results = np.zeros((len(exec_states), 6))
    results[:, 0] = success
    results[:, 1] = invalid
    # TODO: Update other as well

    for _ in range(max_retries):
        # Find the plans that can be run again in closed loop manner
        # Plans are not successful
        # but not invalid (out of bounds or in collision or too long)
        continue_idxs = np.where(~success & ~invalid)[0]
        if len(continue_idxs) == 0:
            break
        print(np.sum(success), np.sum(invalid))
        print(f"{len(continue_idxs)} plans can be run again.")

        # The new start states
        curr_states = []
        for i in continue_idxs:
            curr_states.append(exec_states[i][-1])
        curr_states = np.array(curr_states)

        # Plan for this new start states
        if env == "region":
            plan_states, plan_controls = run_planning_region(
                obj_name,
                model_type,
                learning_type,
                n_data,
                datasets,
                active_sampling,
                predefined_controls=1,
                start_states=curr_states,
                obstacles=[],
                planning_time=replan_time,  # Shorter replanning time
            )
        elif env == "edge":
            plan_states, plan_controls = run_planning_edge(
                obj_name,
                model_type,
                learning_type,
                n_data,
                datasets,
                active_sampling,
                predefined_controls=1,
                start_states=curr_states,
                obstacles=[[0.45, -0.6, 0.04], [0.45, -0.8, 0.04]],
                planning_time=replan_time,  # Shorter replanning time
            )

        # Run the plans again
        res2, success2, invalid2, exec_states2 = run_plans(
            evaluate_fn, plan_states, plan_controls, obj_name, env, 1
        )

        # Update all the results
        success[continue_idxs] = success2
        invalid[continue_idxs] = invalid2
        print(
            f"Current success rate: {np.sum(success)} "
            + f"Current invalid rate: {np.sum(invalid)}"
        )
        results[:, 0] = success
        results[:, 1] = invalid
        # TODO: Update other as well

    # Save results
    results = results.reshape(reps_in_states, -1, 6)
    results = results.mean(axis=1)
    return results, success, invalid, exec_states


if __name__ == "__main__":
    set_seed(42)
    set_ompl_seed(42)

    args = parse_args(
        [
            ("obj_name", "cracker_box_flipped"),
            ("model_type", "residual"),
            ("learning_type", "bait"),
            ("n_data", 100, int),
            ("sampling", 1, int),
        ]
    )
    if "mustard" in args.obj_name:
        env = "region"
    elif "cracker" in args.obj_name:
        env = "edge"
    sampling = "active" if args.sampling else "regular"
    # Skip some of the experiments
    if args.model_type == "residual" and args.sampling == 0:
        exit()
    if args.model_type == "mlp" and args.sampling == 1:
        exit()

    # Load the plans
    name = (
        f"{args.obj_name}_{args.model_type}_{args.learning_type}"
        + f"_{args.n_data}_{sampling}"
    )
    poses_file = f"results/planning/{name}_plan_states.npy"
    controls_file = f"results/planning/{name}_controls.npy"
    plan_states = np.load(poses_file, allow_pickle=True)
    plan_controls = np.load(controls_file, allow_pickle=True)

    # Run closed loop plans
    results, success, invalid, exec_states = run_plans_closed_loop(
        run_plans_pool,  # run_plans_sim
        args.model_type,
        args.learning_type,
        args.n_data,
        args.sampling,
        plan_states,
        plan_controls,
        args.obj_name,
        env,
        reps_in_states=5,
        replan_time=3,
    )
    print(f"{name}: {np.mean(results[:, 0])}")
    np.save(
        f"results/planning/{name}_results_closed_loop.npy",
        results,
    )
    # np.save(
    #     f"results/planning/{name}_exec_states_closed_loop.npy",
    #     np.array(exec_states, dtype=object),
    # )
