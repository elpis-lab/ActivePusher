import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle

from geometry.pose import SE2Pose
from geometry.object_model import get_obj_shape
from utils import DataLoader, parse_args, get_names
from planning.planning_utils import (
    is_edge_success,
    is_state_success,
    plot_states,
)
from planning.planning_utils import in_collision_with_circles

from models.torch_loss_se2 import mse_se2_loss, se2_split_loss
import torch


def main(
    obj_name,
    model_type,
    learning_type,
    n_data,
    active_sampling,
    env,
    verbose=False,
):
    """Run the plans"""
    n_problems = 100
    obstacles = np.array(
        [
            [0.45, -0.6, 0.04],
            [0.45, -0.8, 0.04],
        ]
    )
    circle_poses = obstacles[:, :2]
    circle_rads = obstacles[:, 2]

    # Load initial poses and plans
    sampling = "active" if active_sampling else "regular"
    name = f"{obj_name}_{model_type}_{learning_type}_{n_data}_{sampling}"
    poses_file = f"results/planning/{name}_plan_states.npy"
    controls_file = f"results/planning/{name}_controls.npy"
    plan_states = np.load(poses_file, allow_pickle=True)
    plan_controls = np.load(controls_file, allow_pickle=True)

    in_collision = np.zeros(len(plan_states), dtype=bool)

    # Load model
    model_name, data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")

    # Load control list
    data_loader = DataLoader(data_name)
    datasets = data_loader.load_data()
    control_list = datasets["x_pool"].astype(np.float64)
    delta_list = datasets["y_pool"].astype(np.float64)

    # Plans
    exec_states = []
    delta_poses = []
    pred_delta_poses = []
    # TODO
    results = np.zeros(5)
    for i, (states, controls) in enumerate(zip(plan_states, plan_controls)):
        pose = SE2Pose([states[0][0], states[0][1]], states[0][2])
        e_states = [[*pose.pr()[0], pose.pr()[1]]]

        # Execute the plan
        for j in range(len(controls)):
            control = np.array(controls[j], dtype=np.float64)
            idx = np.argmin(np.linalg.norm(control_list - control, axis=1))
            assert np.allclose(control_list[idx], control)
            delta = delta_list[idx]
            delta_pose = SE2Pose(np.array([delta[0], delta[1]]), delta[2])
            pose = pose @ delta_pose
            pred_delta_pose = SE2Pose(
                [states[j][0], states[j][1]], states[j][2]
            ).invert() @ SE2Pose(
                [states[j + 1][0], states[j + 1][1]], states[j + 1][2]
            )
            delta_poses.append([*delta_pose.pr()[0], delta_pose.pr()[1]])
            pred_delta_poses.append(
                [*pred_delta_pose.pr()[0], pred_delta_pose.pr()[1]]
            )
            inter_state = [*pose.pr()[0], pose.pr()[1]]

            in_collision[i] = (
                in_collision_with_circles(
                    inter_state, obj_shape[:2], circle_poses, circle_rads
                )
                or in_collision[i]
            )

            e_states.append(inter_state)
        exec_states.append(e_states)

        # Plot the states
        if verbose:
            plot_states(e_states, obstacles, states, obj_shape[:2])
    exec_states = np.array(exec_states, dtype=object)

    delta_poses = torch.tensor(delta_poses)
    pred_delta_poses = torch.tensor(pred_delta_poses)
    error = se2_split_loss(pred_delta_poses, delta_poses)
    e = pred_delta_poses[:, :2] - delta_poses[:, :2]
    pos_error = torch.norm(e, dim=1)
    rot_error = torch.abs(
        (pred_delta_poses[:, 2] - delta_poses[:, 2] + torch.pi)
        % (2 * torch.pi)
        - torch.pi
    )
    print(f"Error: {error.mean()}")
    print(f"Position error: {pos_error.mean()}")
    print(f"Rotation error: {rot_error.mean()}")

    # Final state list
    final_states = np.array([states[-1] for states in exec_states])
    final_plan_states = np.array([states[-1] for states in plan_states])

    # Check success
    # Push-to-region Experiment
    if env == "region":
        goal_state = np.array([0, -0.7, 0])
        goal_ranges = np.array([[-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]])
        success = is_state_success(final_states, goal_state, goal_ranges)

    # Push-to-edge Experiment
    elif env == "edge":
        edge = 0.76
        obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
        success = is_edge_success(final_states, obj_shape, edge)
        success = success & ~in_collision

    # Save results
    results[0] = np.sum(success)
    results[1] = error.mean()
    results[2] = pos_error.mean()
    results[3] = rot_error.mean()
    results[4] = np.mean([len(controls) for controls in plan_controls])

    print(f"{name}: {np.sum(success)} / {len(success)}\n")
    np.save(f"results/planning/{name}_results.npy", results)
    np.save(f"results/planning/{name}_exec_states.npy", exec_states)


if __name__ == "__main__":
    # obj_names = [
    #     "mustard_bottle_flipped",
    #     "real_mustard_bottle_flipped",
    # ]
    # envs = ["region"]
    obj_names = [
        "cracker_box_flipped",
        "real_cracker_box_flipped",
    ]
    envs = ["edge"]

    models = ["residual_bait", "mlp_random"]
    n_datas = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # n_datas = [100]
    is_active_sampling = [0, 1]
    for obj_name in obj_names:
        for model in models:
            model_type, learning_type = model.split("_")
            for n_data in n_datas:
                for active_sampling in is_active_sampling:
                    for env in envs:
                        main(
                            obj_name,
                            model_type,
                            learning_type,
                            n_data,
                            active_sampling,
                            env,
                        )
