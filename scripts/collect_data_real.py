import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from geometry.pose import flat_to_matrix, matrix_to_flat
from geometry.random_push import (
    generate_push_params,
    generate_path_from_params,
)
from geometry.object_model import get_obj_shape
from models.physics import push_physics
from real_world.physical_robot import PhysicalUR10
from utils import parse_args
from collect_data import project_se3_pose


########## Hardware Helper Functions ##########
def object_in_bounds(obj_pos, center, bounds):
    """Check if object is within safe bounds."""
    return (
        abs(obj_pos[0] - center[0]) < bounds[0]
        and abs(obj_pos[1] - center[1]) < bounds[1]
    )


def pose_valid(pose):
    """Check if the pose data recevied from the robot is valid"""
    return (
        pose is not None
        and pose.get("pose") is not None
        and isinstance(pose["pose"], np.ndarray)
        and pose["pose"].dtype.kind in {"f", "i"}
    )


def two_step_detect(robot: PhysicalUR10, rough_detect_pose, height=0.12):
    """Check if the vision system detects the object well"""
    adjust_joint_limit(robot)

    # Go to rough detection pose to check the object first
    robot.move_tool(rough_detect_pose)
    obj_pose, bounding_box = get_object_pose(robot, max_retries=0)

    # Approach and check the object again
    robot.move_tool(list(obj_pose[:2, 3]) + [0.35, 0.0, np.pi, 0.0])
    obj_pose, bounding_box = get_object_pose(robot, max_retries=0)

    # Go to the desired height of the object
    robot.move_tool(list(obj_pose[:2, 3]) + [height, 0.0, np.pi, 0.0])
    return obj_pose, bounding_box


def get_object_pose(
    robot: PhysicalUR10,
    max_retries=5,
    rough_detect_pose=None,
    debug_img_id=None,
):
    """Detect the object and check the pose"""
    # Get current robot pose and object pose
    tcp_base_pose = robot.get_ee_transform()
    pose = robot.get_object_pose_hand()

    # In case camera detection is not successful
    c = 0
    while not pose_valid(pose):
        if c >= max_retries - 1 and rough_detect_pose is not None:
            print("(Camera) Moving to rough pose to try last time.")
            two_step_detect(robot, rough_detect_pose, 0.35)
        if c >= max_retries:
            raise ValueError(
                "Camera detection failed. Require manual adjustment!"
            )
        print(f"(Camera) Detection failed, retry {c + 1}/{max_retries}")
        pose = robot.get_object_pose_hand()
        c += 1
        time.sleep(1)

    # Extract the pose
    obj_pose = pose["pose"]
    bounding_box = pose["bounding_box"]
    # Debug
    img = pose["result_image"]
    if img is not None:
        time_stamp = time.time()
        res = "res" if debug_img_id is None else debug_img_id
        img.save(f"debug/{res}_{time_stamp}.jpg")

    # Convert to robot base frame
    tcp_ee_pose = np.eye(4)
    tcp_ee_pose[2, 3] = 0.260
    obj_pose = tcp_base_pose @ np.linalg.inv(tcp_ee_pose) @ obj_pose

    # Project to SE2
    obj_se2 = project_se3_pose(matrix_to_flat(obj_pose))
    # back to 3d pose
    obj_pose = to_se3_matrix(obj_se2)

    # Debug
    np.set_printoptions(precision=5, suppress=True)
    print("Detected Pose: ", obj_se2)
    return obj_pose, bounding_box


def reset_object(robot: PhysicalUR10, obj_pos_xy, push_z, center):
    """Reset the object to the safe position."""
    # Get the direction vector
    dir = center - obj_pos_xy
    dist = np.linalg.norm(dir)
    dir_unit = dir / dist
    angle = np.arctan2(dir[1], dir[0])

    # Pointing downwards, facing pushing direction
    r = R.from_euler("xyz", [-np.pi, 0, angle - np.pi / 2], degrees=False)
    r = r.as_rotvec()

    # Get the position of the pre-push and post-push
    pre_push_dist = 0.15
    pre_push_coord = obj_pos_xy - dir_unit * pre_push_dist
    post_push_coord = obj_pos_xy + dir_unit * dist
    x_d, y_d = pre_push_coord
    x_f, y_f = post_push_coord

    # Move to the recovery push
    print(f"\n(RESET) Push the object from {x_d, y_d} to {x_f, y_f}")
    adjust_joint_limit(robot)
    robot.move_tool([x_d, y_d, 0.10, *r], speed=1.0, acceleration=2.0)
    robot.move_tool([x_d, y_d, push_z, *r])
    robot.move_tool([x_f, y_f, push_z, *r])
    robot.move_tool([x_f, y_f, 0.35, *r], speed=1.0, acceleration=2.0)


def adjust_joint_limit(robot: PhysicalUR10):
    """Adjust the last joint to be within the limit"""
    joint_val = robot.get_q_values()
    if joint_val[5] >= np.pi or joint_val[5] <= -np.pi:
        last_joint_reset = np.array([*joint_val[:5], 0.0])
        robot.move_joint(last_joint_reset, speed=3.0, acceleration=2.0)


########## Other Helper Functions ##########
def to_se3_matrix(se2_pose):
    """Convert SE2 pose to SE3 matrix"""
    se2_pose = np.array(se2_pose)
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler("z", se2_pose[2], degrees=False).as_matrix()
    matrix[:2, 3] = se2_pose[:2]
    return matrix


def get_valid_push_param(
    push_param_candidates, obj_pose, obj_shape, center, bounds
):
    """Generate a valid push param or get it from the candidates"""
    # Get a push param from the candidates
    if push_param_candidates:
        push_param = push_param_candidates.pop(0)
    else:
        push_param = generate_push_params(1)[0]

    # Check if the push param will push the object out of bounds
    while True:
        se2_delta = push_physics(
            push_param[None, :],
            obj_size=obj_shape[:2],
        )[0]
        delta_pose = to_se3_matrix(se2_delta)
        new_pose = obj_pose @ delta_pose
        if object_in_bounds(new_pose[:2, 3], center, bounds):
            break

        push_param_candidates.append(push_param)  # save it for later
        if len(push_param_candidates) > 10:
            print("(Bad Param) Push param candidates are longer than 10.")
        push_param = generate_push_params(1)[0]

    return push_param


def organize_data(obj_name, n_reps, n_data):
    """Re-organize the data"""
    obj_name = f"real_{obj_name}_{n_reps}x{n_data}"
    data_x = np.load("x_" + obj_name + ".npy")
    data_y = np.load("y_" + obj_name + ".npy")

    # Clean unnecessary data_x
    # just to confirm the data is in the correct format
    for i in range(1, n_reps):
        assert np.allclose(
            data_x[i * n_data : (i + 1) * n_data],
            data_x[:n_data],
        )
    data_x = data_x[:n_data]

    # Re-organize data_y
    assert data_y.shape[0] == n_data * n_reps
    data_y = data_y.reshape(n_reps, n_data, -1).transpose(1, 0, 2)

    # Save the re-organized data
    np.save("x_" + obj_name + ".npy", data_x)
    np.save("y_" + obj_name + ".npy", data_y)


########## Main Function ##########
def execute_push(robot: PhysicalUR10, ws_path, push_param):
    """Collect one interaction data"""
    # Adjust the camera angle so that the camera is facing the direction
    for i in range(len(ws_path)):
        r1 = R.from_quat(ws_path[i][[4, 5, 6, 3]])
        r2 = R.from_euler("z", np.pi / 2, degrees=False)
        r = r1 * r2
        ws_path[i] = np.array([*ws_path[i][:3], *r.as_quat()[[3, 0, 1, 2]]])

    # Pre-push Traj and Post-push Traj
    pre_push_pose_flat = ws_path[0].copy()
    pre_push_pose_flat[2] += 0.1
    post_push_pose_flat = ws_path[-1].copy()
    post_push_pose_flat[2] = 0.35

    # Current -> pre-push -> push -> post-push -> Camera position
    adjust_joint_limit(robot)
    robot.move_tool(
        pre_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0
    )
    robot.move_tool(ws_path[0], to_rotvec=True)
    robot.execute_ee_waypoints(ws_path, to_rotvec=True)
    robot.move_tool(
        post_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0
    )

    # Adjust the final camera angle if necessary so that
    # it is more likely to detect the object after pushing
    print("Push Param: ", push_param)
    # if the push can potentially cause big rotation but small translation
    if (push_param[2] > 0.15 and abs(push_param[1]) > 0.25) or (
        push_param[2] > 0.2
    ):
        angle = -np.sign(push_param[1]) * np.pi / 2
        final_r = R.from_quat(post_push_pose_flat[[4, 5, 6, 3]])
        final_r = final_r * R.from_euler("z", angle, degrees=False)
        post_push_pose_flat[3:] = final_r.as_quat()[[3, 0, 1, 2]]
        adjust_joint_limit(robot)
        robot.move_tool(
            post_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0
        )


def main(obj_name, n_data, n_reps=0, detection_wait=0.0, push_offset=0.01):
    """Collect real data"""
    robot = PhysicalUR10()
    execution_dt = 0.008  # UR10 default dt
    tool_offset = np.array([0, 0, -push_offset, 1, 0, 0, 0])

    # Object
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")
    # Robot
    robot_home = np.array([1.3, -1.571, 1.2, -1.2, -1.571, -0.271])
    robot.move_joint(robot_home)

    # Safety bounds for object reset
    center = np.array([0.0, -0.7])
    bounds = np.array([0.3, 0.18])

    # Vision system check
    rough_detect_pose = np.array([-0.05, -0.65, 0.6, 0, np.pi, 0])
    obj_pose, _ = two_step_detect(robot, rough_detect_pose, height=0.15)
    input("Center of The Object?")

    # Data collection setup, always load the existing data
    # This is to prevent something wrong happends
    # in the middle of the real data collection
    if n_reps == 1:
        obj_name = f"real_{obj_name}_{n_data}"
    else:
        obj_name = f"real_{obj_name}_{n_reps}x{n_data}"

    if not os.path.exists("x_" + obj_name + ".npy"):
        data_x = []
        data_y = []
    else:
        data_x = np.load("x_" + obj_name + ".npy").tolist()[:-2]  # :-2
        data_y = np.load("y_" + obj_name + ".npy").tolist()[:-2]  # :-2
    count = len(data_x)
    if count >= n_data * n_reps:
        print(f"Already have {count} data, skip the rest")
        return

    # Start collecting data (regular random buffer)
    # In this case, do not set seed, otherwise the data can be repeated
    if n_reps == 1:
        candidate_push_params = []
    # If running repetitive data collection,
    # set seed for consistent pre-sampled results
    else:
        np.random.seed(42)
        candidate_push_params = generate_push_params(n_data)

    # Always save the data after each interaction
    pbar = tqdm(total=n_data * n_reps, initial=count)
    while count < n_data * n_reps:
        # Check if the object is within the safe bounds to begin with
        while not object_in_bounds(obj_pose[:2, 3], center, bounds):
            reset_object(
                robot, obj_pose[:2, 3], obj_pose[2, 3] + push_offset, center
            )
            # Check if the new object pose
            time.sleep(detection_wait)
            obj_pose, _ = two_step_detect(
                robot, rough_detect_pose, height=0.35
            )

        # Get one push param (regular random)
        if n_reps == 1:
            push_param = get_valid_push_param(
                candidate_push_params, obj_pose, obj_shape, center, bounds
            )
        # Get one push param from the pre-sampled candidates
        else:
            push_param = candidate_push_params[count % n_data]

        # Collect one interaction data
        t_paths, ws_paths = generate_path_from_params(
            matrix_to_flat(obj_pose)[None, :],
            obj_shape,
            push_param[None, :],
            tool_offset=tool_offset,
            pre_push_offset=0.03,
            dt=execution_dt,
        )
        execute_push(robot, ws_paths[0], push_param)

        # Calculate the relative pose
        time.sleep(detection_wait)
        new_obj_pose, _ = get_object_pose(
            robot, 5, rough_detect_pose, debug_img_id=count
        )
        delta_pose = np.linalg.inv(obj_pose) @ new_obj_pose
        se2_delta = project_se3_pose(matrix_to_flat(delta_pose))
        # Save result
        data_x.append(push_param)
        data_y.append(se2_delta)
        np.save("x_" + obj_name + ".npy", np.array(data_x))
        np.save("y_" + obj_name + ".npy", np.array(data_y))

        # The current pose is the new object pose
        obj_pose = new_obj_pose.copy()
        count += 1
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    args = parse_args(
        [
            ("obj_name", "cracker_box_flipped"),
            ("num_data", 2000, int),
            ("n_reps", 1, int),
            ("detection_wait", 0.0, float),
            ("push_offset", 0.01, float),
        ]
    )

    main(
        args.obj_name,
        args.num_data,
        args.n_reps,
        args.detection_wait,
        args.push_offset,
    )

    # # Run this in the end of repetitive data collection
    # organize_data(args.obj_name, args.n_reps, args.num_data)
