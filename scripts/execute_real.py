import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from tqdm import tqdm

from planning.ompl_utils import set_ompl_seed
from planning.planning_utils import plot_states
from geometry.pose import flat_to_matrix, matrix_to_flat
from geometry.random_push import generate_path_from_params
from geometry.object_model import get_obj_shape
from real_world.physical_robot import PhysicalUR10

from utils import DataLoader, set_seed, parse_args, get_names
from collect_data import project_se3_pose
from collect_data_real import two_step_detect, get_object_pose, execute_push
from planning_edge import run_planning as run_planning_edge
from planning_region import run_planning as run_planning_region


def main(
    obj_name,
    model_type,
    learning_type,
    n_data,
    active_sampling,
    predefined_controls,
    env,
    planning_time,
    detection_wait,
    push_offset,
):
    """Execute in real world"""
    # Initialization
    robot = PhysicalUR10()
    execution_dt = 0.008  # UR10 default dt
    tool_offset = np.array([0, 0, -push_offset, 1, 0, 0, 0])
    # Object
    model_name, data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    # Robot
    robot_home = np.array([1.3, -1.571, 1.2, -1.2, -1.571, -0.271])
    robot.move_joint(robot_home)

    # First object pose detection
    rough_detect_pose = np.array([-0.05, -0.65, 0.6, 0, np.pi, 0])
    obj_pose, _ = two_step_detect(robot, rough_detect_pose, height=0.1)
    input("Center of The Object?")

    # Planning
    init_state = project_se3_pose(matrix_to_flat(obj_pose))
    if env == "edge":
        obstacles = np.array([[0.45, -0.6, 0.05], [0.45, -0.8, 0.05]])
        run_planning = run_planning_edge
    else:
        obstacles = np.array([])
        run_planning = run_planning_region
    all_states, all_controls = run_planning(
        obj_name,
        model_type,
        learning_type,
        n_data,
        datasets,
        active_sampling,
        predefined_controls,
        [init_state],
        obstacles,
        planning_time,
    )
    plot_states(all_states[0].astype(np.float64), obstacles, None, obj_shape)

    # Execute the plan (Open-loop)
    controls = all_controls[0].astype(np.float64)
    for count, control in enumerate(controls):
        # Collect one interaction data
        t_paths, ws_paths = generate_path_from_params(
            matrix_to_flat(obj_pose)[None, :],
            obj_shape,
            control[None, :],
            tool_offset=tool_offset,
            pre_push_offset=0.03,
            dt=execution_dt,
        )
        execute_push(robot, ws_paths[0], control)

        # Calculate the relative pose
        time.sleep(detection_wait)
        new_obj_pose, _ = get_object_pose(
            robot, 5, rough_detect_pose, debug_img_id=count
        )
        # delta_pose = np.linalg.inv(obj_pose) @ new_obj_pose
        # se2_delta = project_se3_pose(matrix_to_flat(delta_pose))
        obj_pose = new_obj_pose.copy()


if __name__ == "__main__":
    set_seed(42)
    set_ompl_seed(42)

    args = parse_args(
        [
            # Model
            ("obj_name", "real_mustard_bottle_flipped"),
            ("model_type", "residual"),
            ("learning_type", "bait"),
            ("n_data", "100"),
            # Planning
            ("active_sampling", 1, int),
            ("predefined_controls", 1, int),
            # Execution
            ("detection_wait", 0.0, float),
            ("push_offset", 0.01, float),
        ]
    )

    # Load dataset
    model_name, data_name = get_names(args.obj_name)
    data_loader = DataLoader(data_name)
    datasets = data_loader.load_data()  # Get all data (all in pool)

    # Planning environment
    if "cracker" in args.obj_name:
        env = "edge"
    else:
        env = "region"
    # Planning time
    planning_time = 3

    main(
        args.obj_name,
        args.model_type,
        args.learning_type,
        args.n_data,
        args.active_sampling,
        args.predefined_controls,
        env,
        planning_time,
        args.detection_wait,
        args.push_offset,
    )
