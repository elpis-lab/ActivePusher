import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm

from run_plans import run_plans
from geometry.pose import SE2Pose, euler_to_quat
from planning.planning_utils import out_of_bounds, in_collision_with_circles
from run_plans import run_plans

from simulation.grr_ik import IK
from simulation.mink_ik import UR10IK
from simulation.mujoco_sim import Sim
from collect_data import project_se3_pose
from geometry.random_push import generate_path_from_params


def to_se3_states(se2_states, z):
    """Convert SE2 states to SE3 states"""
    n = se2_states.shape[0]
    pos = np.hstack([se2_states[:, :2], np.ones((n, 1)) * z])
    quat = euler_to_quat(np.hstack([np.zeros((n, 2)), se2_states[:, 2:3]]))
    return np.hstack([pos, quat])


def run_plans_sim(
    obj_name,
    plan_states,
    plan_controls,
    obj_shape,
    circle_poses,
    circle_rads,
    dataset,
    n_envs=20,
):
    """Run the plans in simulation"""
    dt = 0.02
    xml = open("simulation/mujoco_sim.xml").read()
    xml = xml.replace("object_name", obj_name)
    sim = Sim(
        xml,
        n_envs=n_envs,
        robot_joint_dof=6,
        robot_ee_dof=0,
        dt=dt,
        visualize=True,
    )
    # IK solver
    ik = IK("ur10_rod")
    # ik = UR10IK("assets/ur10_rod_ik.xml")

    # Execute the plans
    exec_states = []
    exec_delta_states = []
    states_invalids = []
    n_plan = len(plan_states)
    plan_init_states = np.array([states[0] for states in plan_states])
    plan_lengths = np.array([len(controls) for controls in plan_controls])

    # Run the plan in simulation in parallel
    assert n_plan % n_envs == 0, "n_plan must be divisible by n_envs"
    for i in tqdm(range(n_plan // n_envs)):
        idxs = np.arange(i * n_envs, (i + 1) * n_envs)
        max_step = np.max(plan_lengths[idxs])
        exec_states_round = np.zeros((n_envs, max_step + 1, 3))
        exec_delta_states_round = np.zeros((n_envs, max_step, 3))
        invalid_round = np.zeros((n_envs, max_step + 1), dtype=bool)

        # Reset Robot
        sim.set_robot_init_joints(
            [[np.pi / 2, -1.7, 2, -1.87, -np.pi / 2, np.pi]]
        )

        # Put object in initial states
        inits = plan_init_states[idxs]
        sim.set_obj_init_poses(to_se3_states(inits, obj_shape[2] / 2), 0)
        sim.reset()

        # Execute the plan sequentially
        states = inits
        for step in range(max_step):

            # Prepare push path to execute
            push_params = np.zeros((n_envs, 3))
            for j in range(n_envs):
                if step < plan_lengths[idxs[j]]:
                    push_params[j] = plan_controls[idxs[j]][step]
                else:
                    push_params[j] = np.array([0, 0, -0.05])
            t_paths, ws_paths = generate_path_from_params(
                to_se3_states(states, obj_shape[2] / 2), obj_shape, push_params
            )
            # Solve IK
            pos_waypoints = []
            for j in range(len(ws_paths)):
                traj = ik.ws_path_to_traj(t_paths[j], ws_paths[j])
                waypoints = traj.to_step_waypoints(dt)
                pos_waypoints.append(waypoints)
            # Stack (num_time_step, num_envs, robot_dof) and send it to Sim to execute
            pos_waypoints = np.stack(pos_waypoints, axis=1)

            # Start execution
            delta_poses = sim.execute_waypoints(pos_waypoints, 3.0)
            poses = sim.get_obj_pose(0)[:n_envs]
            deltas = project_se3_pose(delta_poses, axis=[0, 1, 0])
            states = project_se3_pose(poses, axis=[0, 1, 0])
            # Check validity
            invalid_step = out_of_bounds(
                states[:, :2], ((-0.76, 0.76), (-1.1, -0.3))
            ) | in_collision_with_circles(
                states, obj_shape[:2], circle_poses, circle_rads
            )

            # Get results
            exec_delta_states_round[:n_envs, step, :] = deltas
            exec_states_round[:n_envs, step + 1, :] = states
            invalid_round[:n_envs, step + 1] = invalid_step

        # Record data
        for j in range(n_envs):
            exec_states.append(
                exec_states_round[j][: plan_lengths[idxs[j]] + 1].tolist()
            )
            exec_delta_states.append(
                exec_delta_states_round[j][: plan_lengths[idxs[j]]].tolist()
            )
            states_invalids.append(
                invalid_round[j][: plan_lengths[idxs[j]] + 1].tolist()
            )

    sim.close()
    return exec_states, exec_delta_states, states_invalids


if __name__ == "__main__":
    obj_names = [
        "mustard_bottle_flipped",
        "cracker_box_flipped",
    ]

    models = ["residual_bait", "mlp_random"]
    n_datas = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_datas = [100]
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

                    sampling = "active" if active_sampling else "regular"
                    name = (
                        f"{obj_name}_{model_type}_{learning_type}_{n_data}"
                        + f"_{sampling}"
                    )
                    poses_file = f"results/planning/{name}_plan_states.npy"
                    controls_file = f"results/planning/{name}_controls.npy"
                    plan_states = np.load(poses_file, allow_pickle=True)
                    plan_controls = np.load(controls_file, allow_pickle=True)
                    results, success, invalid, exec_states = run_plans(
                        run_plans_sim,
                        plan_states,
                        plan_controls,
                        obj_name,
                        env,
                        reps_in_states=5,
                    )

                    print(f"{name}: {np.mean(results[:, 0])}")
                    print(f"SE2 Error: {np.mean(results[:, 2])}")
                    np.save(f"results/planning/{name}_results.npy", results)
                    np.save(
                        f"results/planning/{name}_exec_states.npy",
                        np.array(exec_states, dtype=object),
                    )
