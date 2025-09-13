import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def push_physics(
    param, obj_size=(0.1, 0.1), relative=True, k_steps=100, push_duration=2
):
    """Calculate the final state of the push given the param."""
    if isinstance(param, np.ndarray):
        param = torch.from_numpy(param)

    # Push parameters, Shape: (N, 1)
    rot, side, distance = param[:, 0:1], param[:, 1:2], param[:, 2:3]
    # if input parameter is relative/normalized value
    if relative:
        # normalized rot values are (0, 0.25, 0.5, 1)
        push_sides = torch.round(rot * 4)
        rot = push_sides * (torch.pi / 2)
        # scale the side to the object size
        mask_odd = push_sides % 2 == 1
        side = side * torch.where(mask_odd, obj_size[0], obj_size[1])

    # Get the velocity at time t, Shape: (K, )
    t = torch.linspace(0, push_duration, k_steps).to(param.device)
    vs, accs = sin_velocity_profile(t, distance, push_duration)

    # Calculate the final state
    x, y, theta = progress_states(rot, side, vs, accs, t[1] - t[0], obj_size)
    x_final = x[:, -1]  # (N,)
    y_final = y[:, -1]  # (N,)
    theta_final = theta[:, -1]  # (N,)

    # Output dim (N, 3)
    return torch.stack([x_final, y_final, theta_final], dim=1)


def sin_velocity_profile(t, d, duration):
    """Get the velocity at time t for a given push distance d and duration."""
    t = t[None, :]  # Shape: (1, K)
    v_max = 2 * d / duration  # Shape: (N, 1)

    # Compute the velocity at time t for each sample given a sin velocity
    # Results have shape (N, K)
    # v = (v_max / 2) * (torch.sin(2 * np.pi * t / duration - np.pi / 2) + 1)
    v = (-v_max / 2) * (torch.cos(2 * torch.pi * t / duration) - 1)
    a = (v_max * torch.pi / duration) * torch.sin(2 * torch.pi * t / duration)

    return v, a


def progress_states(rot, side, velocities, accs, dt, obj_size=(0.1, 0.1)):
    # Model from
    # Manipulation And Active Sensing By Pushing Using Tactile Feedback
    # https://ieeexplore.ieee.org/document/587370
    # Object properties, we assume they are unknown
    c = 0.0187
    # Major simplification to make it work in batch
    # 1, Assume the pusher is always at the same position
    # 2, Assume the push is always perpendicular to the object
    # 3, Assume the push is always in sticky mode
    # 4, The first three steps will definitelyresult in over-pushing,
    #    so we need to clip the push distance to compensate

    # First assume the push is from the right side, then rotate it back later
    cos_rot = torch.cos(rot)
    sin_rot = torch.sin(rot)

    # Adjust the push side dimensions
    # if push from right/left, then size_x is the width, size_y is the height
    # if push from top/bottom, then size_x is the height, size_y is the width
    size = torch.tensor(obj_size, device=rot.device, dtype=rot.dtype)
    size = size.unsqueeze(0).expand(rot.shape[0], 2)
    size_swapped = size[:, [1, 0]]
    swap_mask = (torch.abs(cos_rot) < 1e-3).expand(-1, 2)  # (N, 2)
    size = torch.where(swap_mask, size_swapped, size)
    size_x = size[:, 0].unsqueeze(1)
    size_y = size[:, 1].unsqueeze(1)

    # Push contact point
    x_c = size_x / 2
    y_c = side
    denom = c**2 + x_c**2 + y_c**2
    # Pusher velocity, assume always perpendicular to the object (non-slip)
    vpx = -velocities  # (N, K)
    vpy = 0.0
    # Compute constant body-frame object velocity
    vx = ((c**2 + x_c**2) * vpx + x_c * y_c * vpy) / denom
    vy = ((c**2 + y_c**2) * vpy + x_c * y_c * vpx) / denom
    omega = (x_c * vy - y_c * vx) / (c**2)

    # Compute step-wise state changes
    dx_step = vx * dt
    dy_step = vy * dt
    dtheta = omega * dt
    # Compute cumulative object-frame change
    theta = torch.cumsum(dtheta, axis=1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    dx_local = cos_theta * dx_step - sin_theta * dy_step
    dy_local = sin_theta * dx_step + cos_theta * dy_step

    # Determine where the contact will break
    y_local = torch.cumsum(dy_local, axis=1)
    # Compute the corresponding corner's y coordinates
    # corner point p is [size_x/2, sign(side) * size_y/2]
    p_y = (
        sin_theta * size_x / 2
        + cos_theta * torch.sign(side) * size_y / 2
        + y_local
    )
    # Check when the y coordinates goes beyond y_c
    mask = p_y.abs() <= y_c.abs()  # (N, K)
    mask = mask.cumsum(dim=1) > 1  # disable all contacts after the first one
    dx_local = dx_local.masked_fill(mask, 0.0)
    dy_local = dy_local.masked_fill(mask, 0.0)
    dtheta = dtheta.masked_fill(mask, 0.0)

    # Cumulative local pose
    x_local = torch.cumsum(dx_local, axis=1)
    y_local = torch.cumsum(dy_local, axis=1)
    theta = torch.cumsum(dtheta, axis=1)
    # Adjust position - rotate back to global given the push direction
    x = (x_local * cos_rot) - (y_local * sin_rot)  # (N, K)
    y = (x_local * sin_rot) + (y_local * cos_rot)  # (N, K)

    return x, y, theta


def visualize_process(param, vs, t, x, y, theta, obj_size=(0.1, 0.1)):
    rot, side, distance = (param[0].item(), param[1].item(), param[2].item())
    vs = vs.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()

    def get_corners(x_center, y_center, phi, obj_size):
        half_x = obj_size[0] / 2.0
        half_y = obj_size[1] / 2.0
        local_corners = np.array(
            [
                [-half_x, -half_y],
                [half_x, -half_y],
                [half_x, half_y],
                [-half_x, half_y],
            ]
        )
        # Rotation matrix for angle phi.
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        global_corners = (local_corners @ R.T) + np.array([x_center, y_center])
        return global_corners

    # Set up the plot.
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.grid(True)

    # Plot the path of the object's center.
    ax.plot(x_np, y_np, "k--", label="Center Path")
    # Draw all the square states along the push.
    # Use a slight transparency so overlapping squares can be seen.
    for i in range(len(x_np)):
        # Compute the global orientation for this time step.
        phi = theta_np[i]
        corners = get_corners(x_np[i], y_np[i], phi, obj_size)
        square = patches.Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor="blue",
            linewidth=1,
            alpha=0.3,
        )
        ax.add_patch(square)

    # Draw the push arrow.
    # We assume that the pusher contacts the object at a point defined in the
    if np.abs(np.cos(rot)) < 1e-3:
        contact_local = np.array([obj_size[1] / 2, side])
    else:
        contact_local = np.array([obj_size[0] / 2, side])
    # Rotate the local contact point by the push’s global rotation (rot)
    R_init = np.array(
        [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
    )
    contact_global = R_init @ contact_local
    # We assume the push is applied along the object's local negative x-axis.
    arrow_global = R_init @ np.array([-distance, 0])
    ax.arrow(
        contact_global[0],
        contact_global[1],
        arrow_global[0],
        arrow_global[1],
        head_width=0.02,
        head_length=0.03,
        fc="red",
        ec="red",
        length_includes_head=True,
        label="Push Force",
    )
    # Since we simplify the physics, we also draw the actual push point sequence
    for i in range(len(vs) - 1):
        t_curr = np.eye(3)
        r_curr = np.array(
            [
                [np.cos(rot + theta_np[i]), -np.sin(rot + theta_np[i])],
                [np.sin(rot + theta_np[i]), np.cos(rot + theta_np[i])],
            ]
        )
        t_curr[:2, :2] = r_curr
        t_curr[:2, 2] = np.array([x_np[i], y_np[i]])
        contact_global = t_curr @ np.concatenate([contact_local, [1]])
        dt = t[1] - t[0]
        arrow_global = r_curr @ np.array([-vs[i + 1] * dt, 0])
        ax.arrow(
            contact_global[0],
            contact_global[1],
            arrow_global[0],
            arrow_global[1],
            linewidth=0.1,
            fc="green",
            ec="green",
            length_includes_head=True,
        )

    # Set plot limits with a margin.
    margin = 0.15
    all_x = np.append(x_np, contact_global[0] + arrow_global[0])
    all_y = np.append(y_np, contact_global[1] + arrow_global[1])
    ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
    ax.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
    ax.legend()
    ax.set_title("Static Visualization of a Pushed Square Object")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    # Set up an example push
    push_param = torch.tensor([[0, 0.021, 0.3], [np.pi / 2, 0.04, 0.3]])
    obj_size = (0.1, 0.2)
    push_duration = 2

    # Get states
    rot, side, distance = (
        push_param[:, 0:1],
        push_param[:, 1:2],
        push_param[:, 2:3],
    )
    k_steps = 100
    t = torch.linspace(0, push_duration, k_steps)
    vs, accs = sin_velocity_profile(t, distance, push_duration)
    x, y, theta = progress_states(rot, side, vs, accs, t[1] - t[0], obj_size)
    print("Final state: ", (x[0, -1], y[0, -1], theta[0, -1]))

    # Visualize it
    visualize_process(push_param[0], vs[0], t, x[0], y[0], theta[0], obj_size)
    visualize_process(push_param[1], vs[1], t, x[1], y[1], theta[1], obj_size)
