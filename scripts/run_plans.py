import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from utils import DataLoader, parse_args, get_names
from geometry.pose import SE2Pose
from geometry.object_model import get_obj_shape
from train_model import evaluate_results
from planning.planning_utils import (
    is_edge_success,
    is_state_success,
    plot_states,
)


def get_delta_states(states):
    """Get the intermediate delta states from the states"""
    delta_states = []
    for j in range(len(states) - 1):
        pred_delta_pose = SE2Pose(
            [states[j][0], states[j][1]], states[j][2]
        ).invert() @ SE2Pose(
            [states[j + 1][0], states[j + 1][1]], states[j + 1][2]
        )
        delta_states.append(
            [*pred_delta_pose.pr()[0], pred_delta_pose.pr()[1]]
        )
    return delta_states


def run_plans(
    evaluate_fn,
    obj_name,
    model_type,
    learning_type,
    n_data,
    active_sampling,
    env,
    verbose=False,
):
    """Run the plans"""
    # Environment
    # Push-to-region Experiment
    if env == "region":
        obstacles = np.array([[0.45, -0.6, 0.04], [0.45, -0.8, 0.04]])
        circle_poses = obstacles[:, :2]
        circle_rads = obstacles[:, 2]
    # Push-to-edge Experiment
    elif env == "edge":
        circle_poses = np.array([])
        circle_rads = np.array([])

    # Load data and model
    model_name, data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    data_loader = DataLoader(data_name)
    dataset = data_loader.load_data()

    # Load Plans
    sampling = "active" if active_sampling else "regular"
    name = f"{obj_name}_{model_type}_{learning_type}_{n_data}_{sampling}"
    poses_file = f"results/planning/{name}_plan_states.npy"
    controls_file = f"results/planning/{name}_controls.npy"
    plan_states = np.load(poses_file, allow_pickle=True)
    plan_controls = np.load(controls_file, allow_pickle=True)
    # get plan delta states
    plan_delta_states = []
    for states in plan_states:
        plan_delta_states.append(get_delta_states(states))

    # Execute Plans
    exec_states, exec_delta_states, invalids = evaluate_fn(
        obj_name,
        plan_states,
        plan_controls,
        obj_shape,
        circle_poses,
        circle_rads,
        dataset,
    )
    invalid = [invalid[-1] for invalid in invalids]
    # Plot the results
    if verbose:
        for e_states, states in zip(exec_states, plan_states):
            plot_states(
                e_states, circle_poses, circle_rads, states, obj_shape[:2]
            )

    # Check path deviation
    errors, pos_errors, rot_errors = [], [], []
    for delta_state1, delta_state2 in zip(
        plan_delta_states, exec_delta_states
    ):
        error, pos_error, rot_error = evaluate_results(
            delta_state1, delta_state2
        )
        errors.append(error)
        pos_errors.append(pos_error)
        rot_errors.append(rot_error)
    print(f"Error: {np.mean(errors)}")
    print(f"Position error: {np.mean(pos_errors)}")
    print(f"Rotation error: {np.mean(rot_errors)}")

    # Check success
    success = []
    for states, invalid in zip(exec_states, invalids):
        if env == "region":
            goal_state = np.array([0, -0.7, 0])
            goal_ranges = np.array(
                [[-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]]
            )
            goal_success = is_state_success(states, goal_state, goal_ranges)
        elif env == "edge":
            edge = 0.76
            goal_success = is_edge_success(states, obj_shape, edge)
        success.append(np.any(goal_success & ~invalid))

    # Save results
    results = np.zeros(6)
    results[0] = np.mean(success)
    results[1] = np.mean(invalid)
    results[2] = np.mean(errors)
    results[3] = np.mean(pos_errors)
    results[4] = np.mean(rot_errors)
    results[5] = np.mean([len(controls) for controls in plan_controls])

    print(f"{name}: {np.mean(success)}")
    np.save(f"results/planning/{name}_results.npy", results)
    np.save(
        f"results/planning/{name}_exec_states.npy",
        np.array(exec_states, dtype=object),
    )
    return success, invalid, exec_states
