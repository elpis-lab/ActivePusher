import numpy as np
import time
import os, sys

import mujoco
import mujoco.viewer
from concurrent.futures import ThreadPoolExecutor, wait

from mujoco_sim.mujoco_utils import flat_to_matrix, matrix_to_flat
from mujoco_sim.mujoco_utils import render_mp4


class Sim:
    def __init__(
        self,
        xml,
        n_envs,
        robot_joint_dof,
        robot_ee_dof,
        dt=0.1,
        visualize=True,
        real_time_vis=False,
    ):
        """
        Mujoco Simulation Environment

        This class is implemented in a parallel simulation manner.

        The simulation environment should have one robot and multiple objects.
        qpos is consisting of robot joint positions [:robot_dof]
        and object poses [robot_dof:].
        """
        # Initialize Mujoco
        if "<mujoco" in xml:
            self.mj_model = mujoco.MjModel.from_xml_string(xml)
        else:
            self.mj_model = mujoco.MjModel.from_xml_path(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = None
        if visualize:
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data
            )
            self.viewer.sync()
        self.real_time_vis = real_time_vis

        # Simulation parameters
        # qpos idx
        self.robot_joint_dof = robot_joint_dof
        self.robot_ee_dof = robot_ee_dof
        self.robot_joint_idx = np.arange(robot_joint_dof)
        self.robot_ee_idx = np.arange(
            robot_joint_dof, robot_joint_dof + robot_ee_dof
        )
        obj_idx = np.arange(robot_joint_dof + robot_ee_dof, self.mj_model.nq)
        # TODO, for now it only supports
        # multiple objects with 7DOF or
        # 1 object with any DOF (assume first 7DOF is the object pose)
        if obj_idx.size % 7 == 0:
            self.obj_idxs = obj_idx.reshape(-1, 7)
        else:
            print(
                "Warning: The first object has more than 7DOF. "
                + "The other objects will be ignored in computation."
            )
            self.obj_idxs = obj_idx[None, :7]
        # time
        self.dt = dt
        time_step = self.mj_model.opt.timestep
        self.n_substeps = int(self.dt // time_step)
        assert (
            self.dt % time_step == 0
        ), f"dt must be a multiple of sim time step: {time_step}"

        # Prepare parallelization
        self.n_envs = n_envs
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(n_envs)]
        self.mj_datas_qpos = [mj_data.qpos for mj_data in self.mj_datas]
        self.executor = ThreadPoolExecutor(max_workers=self.n_envs)

        # Store the initial state for reset
        self.init_qpos = np.array(self.mj_datas_qpos)

    def get_sim_info(self):
        """Return simulation infomation"""
        return (self.n_envs, self.dt)

    ########## Parallel Simulation ##########
    def run_sim(self, duration, ctrl=None, thread_fn=None):
        """Run the simulation for a given duration with optional function"""
        if duration <= 0:
            return
        n_steps = int(duration // self.dt)
        self.step_n(n_steps, ctrl, thread_fn)

    def _step_n_thread(
        self, thread_i, n_steps, mj_model, mj_data, ctrl=None, thread_fn=None
    ):
        """Step the simulation for one thread"""
        for j in range(n_steps):
            if ctrl is not None:
                mj_data.ctrl[:] = ctrl[j]
            if thread_fn is not None:
                thread_fn(thread_i, j, mj_model, mj_data)
            for _ in range(self.n_substeps):
                mujoco.mj_step(mj_model, mj_data)

    def step_n(self, n_steps, ctrl=None, thread_fn=None):
        """Step the simulation n times with optional function"""

        def vis_thread_fn(thread_i, j, mj_model, mj_data):
            """Thread function to visualize"""
            if thread_fn is not None:
                thread_fn(thread_i, j, mj_model, mj_data)
            if thread_i == 0:
                self.vis_sync(thread_i)

        if self.viewer:
            fn = vis_thread_fn
        else:
            fn = thread_fn
        futures = [
            self.executor.submit(
                self._step_n_thread,
                i,
                n_steps,
                self.mj_model,
                self.mj_datas[i],
                ctrl,
                fn,
            )
            for i in range(self.n_envs)
        ]
        wait(futures)

    def reset(self, wait_time=0.5):
        """Reset the simulation to the initial state"""
        # Reset is simple and not worth to actually ditribute this
        zero_vel = np.zeros(self.mj_model.nv)
        for i, mj_data in enumerate(self.mj_datas):
            mujoco.mj_resetData(self.mj_model, mj_data)
            mj_data.qpos[:] = self.init_qpos[i]
            mj_data.qvel[:] = zero_vel
            # assume robot joint is position control
            mj_data.ctrl[:] = self.init_qpos[
                i, : self.robot_joint_dof + self.robot_ee_dof
            ]
        # Wait for the simulation to stabilize
        self.run_sim(wait_time)

    def close(self):
        """Close the simulation"""
        if self.viewer:
            self.viewer.close()
        self.executor.shutdown(wait=True)

    ########## Object-related functions ##########
    def set_obj_init_poses(self, init_pose, obj_idx=0, env_idx=None):
        """Set the initial poses of the objects"""
        init_pose, env_idx = self._preprocess_values(init_pose, env_idx)
        self.init_qpos[np.ix_(env_idx, self.obj_idxs[obj_idx])] = init_pose

    def get_obj_pose(self, obj_idx=0, env_idx=None):
        """Return object information"""
        _, env_idx = self._preprocess_values([0], env_idx)
        return np.array(self.mj_datas_qpos)[
            np.ix_(env_idx, self.obj_idxs[obj_idx])
        ]

    ########## Robot-related functions ##########
    def set_robot_init_joints(self, joints, ee_joints=None, env_idx=None):
        """Set the initial joint positions"""
        joints, env_idx = self._preprocess_values(joints, env_idx)
        self.init_qpos[np.ix_(env_idx, self.robot_joint_idx)] = joints
        if ee_joints is not None:
            ee_joints, env_idx = self._preprocess_values(ee_joints, env_idx)
            self.init_qpos[np.ix_(env_idx, self.robot_ee_idx)] = ee_joints

    def get_robot_joints(self, env_idx=None):
        """Get the robot joint positions"""
        _, env_idx = self._preprocess_values([0], env_idx)
        return np.array(self.mj_datas_qpos)[
            np.ix_(env_idx, self.robot_joint_idx)
        ]

    def get_robot_ee(self, env_idx=None):
        """Get the robot end-effector positions"""
        _, env_idx = self._preprocess_values([0], env_idx)
        return np.array(self.mj_datas_qpos)[np.ix_(env_idx, self.robot_ee_idx)]

    def move_ee(self, ee, env_idx=None, wait_time=0.0):
        """Set the robot end-effector positions"""
        self._move_joint(self.robot_ee_idx, ee, env_idx, wait_time)

    def move_arm(self, joints, env_idx=None, wait_time=0.0):
        """Set the robot joint positions"""
        self._move_joint(self.robot_joint_idx, joints, env_idx, wait_time)

    def set_ee(self, ee, env_idx=None):
        """Set the robot end-effector positions"""
        self._set_joint(self.robot_ee_idx, ee, env_idx)

    def set_arm(self, joints, env_idx=None):
        """Set the robot joint positions"""
        self._set_joint(self.robot_joint_idx, joints, env_idx)

    def _move_joint(self, joint_idxs, values, env_idx=None, wait_time=0.0):
        """Move the robot joint positions"""
        values, env_idx = self._preprocess_values(values, env_idx)
        # Set the control
        for i, value in enumerate(values):
            self.mj_datas[env_idx[i]].ctrl[joint_idxs] = value
        self.run_sim(wait_time)

    def _set_joint(self, joint_idxs, values, env_idx=None):
        """Set the robot joint positions"""
        values, env_idx = self._preprocess_values(values, env_idx)
        for i, value in enumerate(values):
            # Set the qpos
            self.mj_datas[env_idx[i]].qpos[joint_idxs] = value
            self.mj_datas[env_idx[i]].qvel[joint_idxs] = 0
            # Set the ctrl
            self.mj_datas[env_idx[i]].ctrl[joint_idxs] = value
            mujoco.mj_forward(self.mj_model, self.mj_datas[env_idx[i]])

    def execute_waypoints(
        self, waypoints, wait_time=0.0, return_intermediate=False
    ):
        """
        Run the push simulation with the given waypoints.
        Args:
            waypoints: the target joint positions at each time step
                       defined as (num_time_step, num_envs, num_joints)
        """
        n_run_steps, n_trials, n_joint = waypoints.shape
        assert n_joint == self.robot_joint_dof, "Invalid joint dimension"
        assert n_trials <= self.n_envs, (
            "required number of execution should not be larger"
            + "than the number of simulation environment"
        )
        env_idx = np.arange(n_trials)
        extra_steps = int(wait_time // self.dt)

        # Save the initial qpos first
        init_qpos = np.array(self.mj_datas_qpos)[:n_trials]
        if return_intermediate:
            intermediate_qpos = np.zeros(
                (1 + n_run_steps + extra_steps, n_trials, init_qpos.shape[1])
            )

        # Define the thread_fn to be passed to parallel_step_n
        def thread_fn(env_i, step_i, mj_model, mj_data):
            """Thread function to be run in parallel"""
            mj_data.ctrl[self.robot_joint_idx] = waypoints[step_i, env_i]
            if return_intermediate:
                intermediate_qpos[step_i, env_i] = mj_data.qpos

        # Start execution
        self.set_arm(waypoints[0], env_idx)
        # Run the sim with the computed trajectory
        self.step_n(n_run_steps, thread_fn=thread_fn)

        # Define the thread_fn to be passed to parallel_step_n
        def thread_stabilize_fn(env_i, step_i, mj_model, mj_data):
            """Thread function to be run in parallel"""
            if return_intermediate:
                intermediate_qpos[step_i, env_i] = mj_data.qpos

        # Run for extra time to stabilize the simulation
        self.step_n(extra_steps, thread_fn=thread_stabilize_fn)

        # Get the last qpos
        last_qpos = np.array(self.mj_datas_qpos)[:n_trials]
        # sim step is run after the thread_fn, store qpos after the last step
        if return_intermediate:
            intermediate_qpos[-1] = last_qpos

        # Compute the relative poses
        init_obj_qpos = init_qpos[:, self.obj_idxs.flatten()]
        last_obj_qpos = last_qpos[:, self.obj_idxs.flatten()]
        relative_qpos = self._get_relative_qpos(init_obj_qpos, last_obj_qpos)

        if return_intermediate:
            return relative_qpos, intermediate_qpos
        else:
            return relative_qpos

    ########## Helper functions ##########
    def _get_relative_qpos(self, init_qpos, last_qpos):
        """Take the inital and last qposand compute the relative 6D poses"""
        relative_qpos = np.zeros_like(init_qpos)
        # For each environment
        for i in range(relative_qpos.shape[0]):
            # For each object
            for j in range(int(relative_qpos.shape[1] // 7)):
                pose1 = flat_to_matrix(init_qpos[i, 7 * j : 7 * (j + 1)])
                pose2 = flat_to_matrix(last_qpos[i, 7 * j : 7 * (j + 1)])
                qpos = matrix_to_flat(np.linalg.inv(pose1) @ pose2)
                relative_qpos[i, 7 * j : 7 * (j + 1)] = qpos
        return relative_qpos

    def _preprocess_values(self, values, env_idx):
        """Preprocess the values and env_idx to match"""
        values = np.asarray(values)

        # Preprocess env_idx first
        # if not provided, use environments that values need
        if env_idx is None:
            size = 1 if values.ndim == 1 else len(values)
            env_idx = np.arange(size)
        # if a single environment provided, convert to array
        if isinstance(env_idx, int):
            env_idx = np.array([env_idx])
        env_idx = np.array(env_idx)

        # Preprocess values
        # if values is 1D, expand it to n_envs
        if values.ndim == 1:
            values = np.tile(values, (len(env_idx), 1))
        else:
            assert len(values) == len(
                env_idx
            ), "Values need to be 1D or have the same length as env_idx"

        return values, env_idx

    def vis_sync(self, env_idx=0):
        """Sync the simulation state to the viewer"""
        if self.viewer is None:
            return
        self.mj_data.qpos[:] = self.mj_datas_qpos[env_idx]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.viewer.sync()
        if self.real_time_vis:
            time.sleep(self.dt)

    def render_state(self, state, filename):
        """Render the state qpos (n_frame, nq) into a mp4 video"""
        render_mp4(self.mj_model, self.mj_data, state, self.dt, filename)


def test(sim: Sim):
    # Testing
    sim.set_robot_init_joints(
        np.array([np.pi / 2, -1.7, 2, -1.87, -np.pi / 2, np.pi])
    )
    sim.set_robot_init_joints(
        np.array([1.620, -2.480, 2.265, -1.350, -1.575, 0.0])
    )
    sim.set_obj_init_poses(np.array([0, -0.7, 0, 1, 0, 0, 0]), 0)
    sim.reset()
    input("Start test")

    # Test control
    ctrl = np.array([-1.5, -1.5, 1.5, -1.5, -1.5, 0])
    sim.move_arm(ctrl, wait_time=1.0)

    # Test waypoints
    waypoints = np.linspace(ctrl - 0.5, ctrl + 0.5, 30)
    waypoints = waypoints[:, None, :].repeat(sim.n_envs, axis=1)
    relative_qpos, intermediate_qpos = sim.execute_waypoints(
        waypoints, wait_time=1.0, return_intermediate=True
    )
    print(relative_qpos[0], relative_qpos.shape)
    print(
        intermediate_qpos[0, 0],
        intermediate_qpos[-1, 0],
        intermediate_qpos.shape,
    )

    # Test getters
    print(sim.get_sim_info())
    print(sim.get_robot_joints()[0], sim.get_robot_joints().shape)
    print(sim.get_obj_pose()[0], sim.get_obj_pose().shape)

    input("Test done")
    sim.close()


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(curr_dir)

    xml = open(os.path.join(curr_dir, "mujoco_sim.xml")).read()
    xml = xml.replace("object_name", "cracker_box_flipped")
    sim = Sim(
        xml,
        n_envs=20,
        robot_joint_dof=6,
        robot_ee_dof=0,
        dt=0.02,
        visualize=True,
    )
    test(sim)
