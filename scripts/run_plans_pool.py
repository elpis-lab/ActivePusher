import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from geometry.pose import SE2Pose
from planning.planning_utils import out_of_bounds, in_collision_with_circles
from run_plans import run_plans


def run_plans_pool(
    obj_name,
    plan_states,
    plan_controls,
    obj_shape,
    circle_poses,
    circle_rads,
    dataset,
):
    """
    Run the plans in a simplified manner

    If planned with pre-defined control list,
    a very large batch of control parameters from the dataset,
    which is NOT used for training the model.

    In this case, we can assume that the actual effect of these controls
    are the same as in dataset. We can therefore evaluate the plan using
    only the corresponding dataset values without running the simulation again.
    """
    # load potential control list and delta list
    control_list = dataset["x_pool"].astype(np.float64)
    delta_list = dataset["y_pool"].astype(np.float64)

    # Execute the plans
    exec_states = []
    exec_delta_states = []
    invalids = []
    # for each plan
    for i, (states, controls) in enumerate(zip(plan_states, plan_controls)):
        # Start state
        pose = SE2Pose([states[0][0], states[0][1]], states[0][2])
        e_states = [[*pose.pr()[0], pose.pr()[1]]]
        e_delta_states = []
        invalid = [False]

        # Find corresponding effect of plan controls
        controls = np.asarray(controls, dtype=np.float64)
        dists = np.linalg.norm(
            controls[:, None, :] - control_list[None, :, :], axis=2
        )
        idxs = np.argmin(dists, axis=1)
        assert np.allclose(control_list[idxs], controls)
        deltas = np.asarray(delta_list)[idxs]

        # Execute the plan
        for delta in deltas:
            # update pose
            delta_pose = SE2Pose(np.array([delta[0], delta[1]]), delta[2])
            pose = pose @ delta_pose
            e_delta_states.append([*delta_pose.pr()[0], delta_pose.pr()[1]])
            e_state = [*pose.pr()[0], pose.pr()[1]]
            e_states.append(e_state)

            # check validity
            invalid.append(
                in_collision_with_circles(
                    e_state, obj_shape[:2], circle_poses, circle_rads
                )
                or out_of_bounds(
                    e_state[:2], pos_range=((-0.76, 0.76), (-1.1, -0.3))
                )
            )

        exec_states.append(e_states)
        exec_delta_states.append(e_delta_states)
        invalids.append(invalid)

    return exec_states, exec_delta_states, invalids


if __name__ == "__main__":
    obj_names = [
        "mustard_bottle_flipped",
        "cracker_box_flipped",
        "real_mustard_bottle_flipped",
        "real_cracker_box_flipped",
    ]

    models = ["residual_bait", "mlp_random"]
    n_datas = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    is_active_sampling = [0, 1]
    for obj_name in obj_names:
        if "mustard" in obj_name:
            env = "region"
        elif "cracker" in obj_name:
            env = "edge"
        for model in models:
            model_type, learning_type = model.split("_")

            for n_data in n_datas:
                for active_sampling in is_active_sampling:
                    run_plans(
                        run_plans_pool,
                        obj_name,
                        model_type,
                        learning_type,
                        n_data,
                        active_sampling,
                        env,
                    )
