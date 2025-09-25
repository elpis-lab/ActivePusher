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
    plan_states,
    plan_controls,
    obj_name,
    env,
    reps_in_states=1,
    verbose=False,
):
    """Run the plans"""
    # Environment
    # Push-to-region Experiment
    if env == "region":
        obstacles = np.array([])
        circle_poses = np.array([])
        circle_rads = np.array([])
    # Push-to-edge Experiment
    elif env == "edge":
        obstacles = np.array([[0.45, -0.6, 0.04], [0.45, -0.8, 0.04]])
        circle_poses = obstacles[:, :2]
        circle_rads = obstacles[:, 2]

    # Load data and model
    model_name, data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    data_loader = DataLoader(data_name)
    dataset = data_loader.load_data()

    # get plan delta states
    plan_delta_states = []
    for states in plan_states:
        plan_delta_states.append(get_delta_states(states))

    # Execute Plans
    exec_states, exec_delta_states, states_invalids = evaluate_fn(
        obj_name,
        plan_states,
        plan_controls,
        obj_shape,
        circle_poses,
        circle_rads,
        dataset,
    )

    invalid = np.array([invalid[-1] for invalid in states_invalids])
    # Plot the results
    if verbose:
        for e_states, states in zip(exec_states, plan_states):
            plot_states(e_states, obstacles, states, obj_shape[:2])

    # Check path deviation
    errors = np.zeros(len(exec_states))
    pos_errors = np.zeros(len(exec_states))
    rot_errors = np.zeros(len(exec_states))
    for i, (delta_state1, delta_state2) in enumerate(
        zip(plan_delta_states, exec_delta_states)
    ):
        error, pos_error, rot_error = evaluate_results(
            delta_state1, delta_state2
        )
        errors[i] = error
        pos_errors[i] = pos_error
        rot_errors[i] = rot_error
    print(f"Error: {np.mean(errors)}")
    print(f"Position error: {np.mean(pos_errors)}")
    print(f"Rotation error: {np.mean(rot_errors)}")

    # Check success and validity
    success = np.zeros(len(exec_states), dtype=bool)
    for i, (states, inv) in enumerate(zip(exec_states, states_invalids)):
        if env == "region":
            goal_state = np.array([0, -0.7, 0])
            goal_ranges = np.array(
                [[-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]]
            )
            goal_success = is_state_success(states, goal_state, goal_ranges)
        elif env == "edge":
            edge = 0.76
            goal_success = is_edge_success(states, obj_shape, edge)

        # If the goal is reached during execution
        first_goal_idx = np.argmax(goal_success)
        if goal_success.any():
            success[i] = not np.any(inv[: first_goal_idx + 1])
        else:
            success[i] = False
        if success[i]:
            invalid[i] = False  # clear all invalid if success before
        else:
            invalid[i] = np.any(inv)  # whole plan is invalid if any invalid

    # Save results
    results = np.zeros((len(exec_states), 6))
    results[:, 0] = success
    results[:, 1] = invalid
    results[:, 2] = errors
    results[:, 3] = pos_errors
    results[:, 4] = rot_errors
    results[:, 5] = np.array([len(controls) for controls in plan_controls])
    # split if reps in states is not 1
    results = results.reshape(reps_in_states, -1, 6)
    results = results.mean(axis=1)
    return results, success, invalid, exec_states
