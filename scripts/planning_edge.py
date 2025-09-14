import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle

from geometry.object_model import get_obj_shape
from geometry.pose import angle_diff
from train_model import load_model, get_push_physics
from utils import DataLoader, set_seed, parse_args, get_names

from planning.ompl_utils import set_ompl_seed
from planning.ompl_utils import SE2ControlPlanner
from planning.planning_utils import get_random_se2_states


def generate_initial_states(n_states):
    """Generate initial states"""
    states = get_random_se2_states(n_states, pos_range=(0.1, 0.2, -0.6, -0.8))
    np.save("data/planning_edge_initial_states.npy", states)


def run_planning(
    obj_name,
    model_type,
    learning_type,
    n_data,
    datasets,
    active_sampling,
    predefined_controls,
    start_states,
    obsatcles,
    planning_time,
    n_reps=5,
):
    """Main function to do planning"""
    # Planning parameters
    # state bounds
    bounds = ((-0.76, 0.76), (-1.1, -0.3))
    # control bounds
    control_bounds = ((0, 4), (-0.4, 0.4), (0.0, 0.3))

    # Load dynamics model
    m_id = 0
    model_name, data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    equation = get_push_physics(model_type, obj_shape[:2])
    model = load_model(model_type, equation, 0)
    name = f"{obj_name}_{model_type}_{learning_type}_{m_id}_{n_data}"
    model.load(f"results/models/{name}.pth")
    # Load data that is used for training, for active planning
    used_indices = pickle.load(
        open(f"results/learning/idx_used_{obj_name}_{model_type}.pkl", "rb")
    )
    used_indices = used_indices[learning_type][m_id][: int(n_data)]
    x_train = datasets["x_pool"][used_indices]
    # Planning control lists
    if predefined_controls:
        # samplable controls, exclude training data from control pool
        mask = np.ones(len(datasets["x_pool"]), dtype=bool)
        mask[used_indices] = False
        controls = datasets["x_pool"][mask].astype(np.float64)
    else:
        controls = None

    # Define commom goal
    goal_state = np.array([0.72, -0.7, 0])
    goal_ranges = np.array(
        [[-0.015, 0.015], [-0.3, 0.3], [-np.pi / 12, np.pi / 12]]
    )

    # Instantiate the planner
    planner = SE2ControlPlanner(
        bounds,
        control_bounds,
        obj_shape,
        obsatcles,
        model,
        x_train,
        active_sampling,
        controls,
    )

    # Define the start and goal
    all_controls = []  # [n_reps x n_problems, n_steps]
    all_states = []  # [n_reps x n_problems, n_steps]

    # TODO
    n_problems = 100
    for _ in range(n_reps):
        for start_state in start_states[:n_problems]:

            # Change goal state based on start state
            rot_goals = [-np.pi / 2, np.pi / 2]
            rot_diff = [angle_diff(start_state[2], g) for g in rot_goals]
            goal_state[2] = rot_goals[np.argmin(rot_diff)]

            # Plan
            plan_states, plan_controls = planner.plan(
                start_state,
                goal_state,
                goal_ranges,
                planning_time,
                accept_approximate=False,
            )
            if not plan_controls:
                plan_states = [start_state, start_state]
                plan_controls = [[0, 0, -0.05]]
            all_states.append(plan_states)
            all_controls.append(plan_controls)
    all_states = np.array(all_states, dtype=object)
    all_controls = np.array(all_controls, dtype=object)

    return all_states, all_controls


if __name__ == "__main__":
    set_seed(42)
    set_ompl_seed(42)
    # generate_initial_states(100)

    args = parse_args(
        [
            # Model
            ("obj_name", "cracker_box_flipped"),
            ("model_type", "residual"),
            ("learning_type", "bait"),
            ("n_data", "100"),
            # Planning
            ("active_sampling", 1, int),
            ("predefined_controls", 1, int),
            ("n_reps", 5, int),
        ]
    )

    # Load dataset
    model_name, data_name = get_names(args.obj_name)
    data_loader = DataLoader(data_name)
    datasets = data_loader.load_data()  # Get all data (all in pool)
    # Load problem set
    start_states = np.load("data/planning_edge_initial_states.npy")

    # Planning environment
    obstacles = [
        [0.45, -0.6, 0.05],
        [0.45, -0.8, 0.05],
    ]
    # Planning time
    planning_time = 3

    all_states, all_controls = run_planning(
        args.obj_name,
        args.model_type,
        args.learning_type,
        args.n_data,
        datasets,
        args.active_sampling,
        args.predefined_controls,
        start_states,
        obstacles,
        planning_time,
        args.n_reps,
    )

    # Save the results
    sampling = "active" if args.active_sampling else "regular"
    name = (
        f"{args.obj_name}_{args.model_type}_{args.learning_type}"
        + f"_{args.n_data}_{sampling}"
    )
    np.save(f"results/planning/{name}_plan_states.npy", all_states)
    np.save(f"results/planning/{name}_controls.npy", all_controls)
