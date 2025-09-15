import numpy as np
import matplotlib.pyplot as plt


# State Utils
def get_random_se2_states(
    n_data, pos_range=((-0.2, 0.2), (-0.9, -0.5)), euler_range=(-np.pi, np.pi)
):
    """Generate random initial states"""
    pos_x = np.random.uniform(pos_range[0][0], pos_range[0][1], (n_data, 1))
    pos_y = np.random.uniform(pos_range[1][0], pos_range[1][1], (n_data, 1))
    euler = np.random.uniform(euler_range[0], euler_range[1], (n_data, 1))
    states = np.concatenate([pos_x, pos_y, euler], axis=-1)
    return states


# Validity Check
def out_of_bounds(obj_pos, pos_range=((-0.76, 0.76), (-1.1, -0.3))):
    """Check if the object is out of bounds"""
    obj_pos = np.array(obj_pos)
    single = obj_pos.ndim == 1
    if single:
        obj_pos = obj_pos[None, :]

    # Check bounds
    lower = np.array([r[0] for r in pos_range])
    upper = np.array([r[1] for r in pos_range])
    outs = np.any((obj_pos < lower) | (obj_pos > upper), axis=1)

    if single:
        return outs[0]
    return outs


def in_collision_with_circles(
    obj_pose, obj_shape, circle_poses, circle_radius
):
    """Check if the object is in collision with a list of circles"""
    if len(circle_poses) == 0 or len(circle_radius) == 0:
        return False

    obj_pose = np.array(obj_pose)
    single = obj_pose.ndim == 1
    if single:
        obj_pose = obj_pose[None, :]

    x, y, yaw = obj_pose[:, 0], obj_pose[:, 1], obj_pose[:, 2]
    w, h = obj_shape[:2]
    hx, hy = 0.5 * w, 0.5 * h

    # Translate circle centers into object frame
    dx = circle_poses[:, 0][None, :] - x[:, None]
    dy = circle_poses[:, 1][None, :] - y[:, None]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    lx = c * dx + s * dy
    ly = -s * dx + c * dy

    # Find closest point on the object to the circle center
    # Clamp each circle center to object extents
    qx = np.clip(lx, -hx, hx)
    qy = np.clip(ly, -hy, hy)

    # Distance squared from circle center to nearest point on object
    ddx = lx - qx
    ddy = ly - qy
    dist2 = ddx * ddx + ddy * ddy

    # Collision check
    circle_radius.reshape((-1, circle_radius.shape[0]))
    hits = np.any(dist2 <= circle_radius**2, axis=1)

    # Return result
    if single:
        return hits[0]
    return hits


def get_box_corners(poses, obj_shape):
    """Get the corners of the box of given poses"""
    poses = np.atleast_2d(poses)
    n_data = poses.shape[0]
    w, h = obj_shape[:2]

    # Local box corners
    local_corners = np.array(
        [
            [-w / 2, -h / 2, 1.0],
            [-w / 2, h / 2, 1.0],
            [w / 2, -h / 2, 1.0],
            [w / 2, h / 2, 1.0],
        ]
    ).T

    # Transforms
    sin_yaw = np.sin(poses[:, 2])
    cos_yaw = np.cos(poses[:, 2])
    h_matrices = np.tile(np.eye(3)[None], (n_data, 1, 1))
    h_matrices[:, :2, :2] = np.stack(
        (cos_yaw, -sin_yaw, sin_yaw, cos_yaw), axis=1
    ).reshape(-1, 2, 2)
    h_matrices[:, :2, 2] = poses[:, :2]

    # Transform corners
    corners = h_matrices @ local_corners
    corners = corners[:, :2, :].transpose(0, 2, 1)
    return corners


def is_edge_success(poses, obj_shape, edge=0.76, threshold=0.025):
    """Check if the push-to-edge plan is successful"""
    poses = np.atleast_2d(poses)
    xs = poses[:, 0]
    corners = get_box_corners(poses, obj_shape)  # (N, 4, 2)
    max_xs = np.max(corners[:, :, 0], axis=1)
    # check if the object has enough space to be grasped
    success = (max_xs - edge >= threshold) & (xs < edge)
    return success


def is_state_success(poses, goal, goal_region):
    """Check if the state is successful"""
    poses = np.atleast_2d(poses)
    goal = np.asarray(goal)
    goal_region = np.asarray(goal_region)

    # Check if it is in the goal region
    d = poses - goal
    d[:, 2] = (d[:, 2] + np.pi) % (2.0 * np.pi) - np.pi  # wrap to pi
    in_x = (goal_region[0, 0] <= d[:, 0]) & (d[:, 0] <= goal_region[0, 1])
    in_y = (goal_region[1, 0] <= d[:, 1]) & (d[:, 1] <= goal_region[1, 1])
    in_yaw = (goal_region[2, 0] <= d[:, 2]) & (d[:, 2] <= goal_region[2, 1])
    return in_x & in_y & in_yaw


# Plotting
def plot_states(
    states,
    circle_poses=(),
    circle_rads=(),
    planned_states=None,
    obj_shape=None,
):
    """Plot the states of the object."""
    states = np.array(states)
    if planned_states is not None:
        planned_states = np.array(planned_states)

    plt.figure(figsize=(8, 6))
    # Plot the table as the background
    draw_rectangle(0, -0.505, 1.524, 1.524, 0, "k", alpha=0.1)
    # Plot a robot
    draw_rectangle(0, 0, 0.2, 0.2, 0, "gray", alpha=1.0, label="Robot")
    # Plot the obstacles
    for circle_pose, circle_rad in zip(circle_poses, circle_rads):
        draw_circle(
            circle_pose[0],
            circle_pose[1],
            circle_rad,
            "r",
            alpha=0.3,
            label="Obstacle",
        )

    # Plot the states path
    if planned_states is not None:
        plt.plot(
            planned_states[:, 0],
            planned_states[:, 1],
            "o-",
            color="b",
            label="Planned Path",
        )
    plt.plot(
        states[:, 0],
        states[:, 1],
        "o-",
        color="g",
        label="Actual Path",
    )
    # If object shape is provided, draw rectangles for start and goal
    if obj_shape is not None:
        w, l = obj_shape[0], obj_shape[1]

        # Draw rectangles
        if planned_states is not None:
            for state in planned_states:
                state_x, state_y, state_theta = state
                draw_rectangle(
                    state_x, state_y, w, l, state_theta, "b", alpha=0.3
                )
        for state in states:
            state_x, state_y, state_theta = state
            draw_rectangle(state_x, state_y, w, l, state_theta, "g", alpha=0.3)

    # Plot start positions
    plt.plot(states[0, 0], states[0, 1], "ro", label="Start")

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Push Path")
    plt.legend()
    plt.show()


def draw_circle(x, y, r, color="r", alpha=0.3, label=None):
    """Draw a circle at the given position with the given radius."""
    circle = plt.Circle((x, y), r, color=color, alpha=alpha)
    plt.gca().add_patch(circle)
    if label is not None:
        plt.text(x, y, label, ha="center", va="center", color="black")


def draw_rectangle(
    x, y, width, length, theta, color="b", alpha=0.5, label=None
):
    """Draw a rectangle at the given position with the given orientation."""
    # Calculate the four corners of the rectangle
    corners = np.array(
        [
            [-width / 2, -length / 2],
            [width / 2, -length / 2],
            [width / 2, length / 2],
            [-width / 2, length / 2],
            [-width / 2, -length / 2],  # Close the rectangle
        ]
    )

    # Rotate the corners
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotated_corners = np.dot(corners, rot_matrix.T)
    # Translate the corners
    translated_corners = rotated_corners + np.array([x, y])

    # Plot the rectangle
    plt.plot(
        translated_corners[:, 0],
        translated_corners[:, 1],
        color=color,
        alpha=alpha,
    )
    plt.fill(
        translated_corners[:, 0],
        translated_corners[:, 1],
        color=color,
        alpha=alpha,
    )

    # Add text label
    if label is not None:
        plt.text(x, y, label, ha="center", va="center", color="black")
