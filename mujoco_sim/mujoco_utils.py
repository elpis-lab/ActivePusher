import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def render_mp4(
    mj_model,
    mj_data,
    qpos_sequence,
    dt,
    file_path,
    height=720,
    width=1280,
    scene_option_flags={},
):
    """
    Render a sequence of qpos data into mp4 video

    qpos_sequence should be a list of qpos data that represents
    the qpos data at each time step
    """
    import imageio

    # Initialize renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    for flag, value in scene_option_flags.items():
        scene_option.flags[flag] = value

    # Render frames
    mujoco.mj_resetData(mj_model, mj_data)
    frames = []
    for qpos in qpos_sequence:
        mj_data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

    # Save frames to mp4
    frame_rate = 1 / dt
    with imageio.get_writer(file_path, fps=frame_rate) as writer:
        for frame in frames:
            writer.append_data(frame)

    return frames


def flat_to_matrix(flat):
    """Convert a flat 7D array to a 4x4 homogeneous matrix"""
    flat = np.array(flat)
    single = False
    if flat.ndim == 1:
        flat = flat[None, :]
        single = True

    # Convert to matrices
    positions = flat[:, :3]
    quat_xyzw = flat[:, [4, 5, 6, 3]]
    rotations = R.from_quat(quat_xyzw).as_matrix()
    matrices = np.tile(np.eye(4), (flat.shape[0], 1, 1))
    matrices[:, :3, 3] = positions
    matrices[:, :3, :3] = rotations

    if single:
        return matrices[0]
    return matrices


def matrix_to_flat(matrix):
    """Convert a 4x4 homogeneous matrix to a flat 7D array"""
    matrix = np.array(matrix)
    single = False
    if matrix.ndim == 2:
        matrix = matrix[None, :, :]
        single = True

    # Convert to flat
    positions = matrix[:, :3, 3]
    quat_xyzw = R.from_matrix(matrix[:, :3, :3]).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    flats = np.concatenate([positions, quat_wxyz], axis=-1)

    if single:
        return flats[0]
    return flats
