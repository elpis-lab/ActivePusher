import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from geometry.pose import euler_to_quat

# from simulation.grr_ik import IK
from simulation.mink_ik import UR10IK
from geometry.random_push import (
    get_random_push,
    generate_push_params,
    generate_path_from_params,
)
from geometry.object_model import get_obj_shape
from utils import parse_args, set_seed

from simulation.mujoco_sim import Sim


def execute_push(sim: Sim, ik, init_state, t_path, ws_path, dt):
    """Execute push and return SE2 pose"""
    # Solve IK
    pos_waypoints = []
    for j in range(len(ws_path)):
        traj = ik.ws_path_to_traj(t_path[j], ws_path[j])
        waypoints = traj.to_step_waypoints(dt)
        pos_waypoints.append(waypoints)
    # Stack (num_time_step, num_envs, robot_dof) and send it to Sim to execute
    pos_waypoints = np.stack(pos_waypoints, axis=1)

    # Start execution
    sim.set_obj_init_poses(init_state, 0)
    sim.reset()
    poses = sim.execute_waypoints(pos_waypoints, 3.0)
    se2_pose = project_se3_pose(poses, axis=[0, 1, 0])
    return se2_pose


def collect_data(obj_name, n_data, random_init=True, push_params=None):
    """Collect n_data for obj_name"""
    # Sim class
    xml = open("simulation/mujoco_sim.xml").read()
    xml = xml.replace("object_name", obj_name)
    sim = Sim(
        xml,
        n_envs=20,
        robot_joint_dof=6,
        robot_ee_dof=0,
        dt=0.02,
        visualize=True,  # False
        real_time_vis=False,
    )
    # IK solver
    # ik = IK("ur10_rod")
    ik = UR10IK("assets/ur10_rod_ik.xml")

    # Initial state parameters
    n_envs, dt = sim.get_sim_info()
    sim.set_robot_init_joints([[np.pi / 2, -1.7, 2, -1.87, -np.pi / 2, np.pi]])
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")

    # Data check and initialize container
    assert n_data > 0 and (n_data < n_envs or n_data % n_envs == 0)
    n_rounds = max(int(n_data // n_envs), 1)
    results = np.zeros((n_data, 3))

    # Get initial object poses
    if random_init:
        init_states = get_random_se2_states(n_data, obj_shape[2] / 2)
    else:
        init_states = np.tile(
            [0, -0.7, obj_shape[2] / 2, 1, 0, 0, 0], (n_data, 1)
        )
    # Compute push parameters if not pre-provided
    if push_params is None:
        push_params, t_paths, ws_paths = get_random_push(
            n_data, init_states, obj_shape
        )
    else:
        assert len(push_params) == n_data
        t_paths, ws_paths = generate_path_from_params(
            init_states, obj_shape, push_params
        )

    # Start collectin
    for i in tqdm(range(n_rounds)):
        # Push parameters for this round
        init_state = init_states[n_envs * i : n_envs * (i + 1)]
        t_path = t_paths[n_envs * i : n_envs * (i + 1)]
        ws_path = ws_paths[n_envs * i : n_envs * (i + 1)]

        # Execute this round
        se2_pose = execute_push(sim, ik, init_state, t_path, ws_path, dt)
        results[i * n_envs : (i + 1) * n_envs] = se2_pose

    sim.close()
    return push_params, results


def collect_repetitive_data(obj_name, n_data, n_reps, random_init=True):
    """Collect n_reps * n_data for obj_name"""
    # Generate random push waypoints to repeat
    push_params = generate_push_params(n_data)
    results_rep = np.zeros((n_data, n_reps, 3))

    # Start collecting
    for r in range(n_reps):
        _, results = collect_data(obj_name, n_data, random_init, push_params)
        results_rep[:, r, :] = results

    return push_params, results_rep


########## Helper functions ##########
def get_random_se2_states(
    n_envs,
    z=0,
    pos_range=((-0.2, 0.2), (-0.9, -0.5)),
    euler_range=(-np.pi, np.pi),
):
    """Get random SE2 states for n_envs"""
    pos_x = np.random.uniform(pos_range[0][0], pos_range[0][1], (n_envs, 1))
    pos_y = np.random.uniform(pos_range[1][0], pos_range[1][1], (n_envs, 1))
    pos_z = z * np.ones((n_envs, 1))
    pos = np.concatenate([pos_x, pos_y, pos_z], axis=-1)

    euler = np.random.uniform(euler_range[0], euler_range[1], n_envs)
    quat = euler_to_quat([(0, 0, e) for e in euler])

    return np.concatenate([pos, quat], axis=-1)


def project_se3_pose(poses, axis=[0, 1, 0]):
    """Project SE3 pose to SE2"""
    poses = np.array(poses)
    single = poses.ndim == 1
    if single:
        poses = poses[None, :]

    # Position
    xy = poses[:, :2]
    # Rotation
    # assuming axis is the forward axis that is always
    # parallel to the table direction
    # find the angle difference for relative rotation
    rotations = R.from_quat(poses[:, [4, 5, 6, 3]])
    axis0 = np.tile(axis, (poses.shape[0], 1))
    axis1 = rotations.apply(axis0)
    # dot and perp‐cross in 2D
    dots = np.sum(axis0[:, :2] * axis1[:, :2], axis=1)
    cross_zs = axis0[:, 0] * axis1[:, 1] - axis0[:, 1] * axis1[:, 0]
    # signed angle = atan2(sin, cos)
    yaw = np.arctan2(cross_zs, dots)[:, None]
    se2_poses = np.concatenate([xy, yaw], axis=-1)

    if single:
        return se2_poses[0]
    return se2_poses


if __name__ == "__main__":
    args = parse_args([("obj_name", "cracker_box_flipped")])
    os.makedirs("data", exist_ok=True)
    set_seed(42)

    # Collect data
    n_data = 10000
    push_params, results = collect_data(args.obj_name, n_data)
    np.save(f"data/x_{args.obj_name}_{n_data}.npy", push_params)
    np.save(f"data/y_{args.obj_name}_{n_data}.npy", results)
