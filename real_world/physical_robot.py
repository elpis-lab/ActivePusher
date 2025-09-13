import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from real_world.rtde import RTDE
from real_world.gripper import Gripper
from real_world.camera import Camera


class PhysicalUR10:
    def __init__(self):
        """Initialize the physical UR10 robot class"""
        self.rtde = RTDE("192.168.0.100")
        self.gripper = Gripper("192.168.0.101", "8005")
        self.top_cam = Camera("192.168.0.101", "5001")
        self.hand_cam = Camera("192.168.0.101", "5000")

    # Joint control
    def execute_trajectory(self, waypoints, d_t: float = 0.008, **kwargs):
        """Execute a trajectory"""
        # speed_list = []

        # Execute each waypoint
        for waypoint in waypoints:
            start_t = self.rtde.rtde_c.initPeriod()
            self.rtde.servo_joint(waypoint, time=d_t, **kwargs)
            self.rtde.rtde_c.waitPeriod(start_t)
            # speed_list.append(self.get_ee_speed())

        # Stop servo
        self.rtde.rtde_c.servoStop()
        time.sleep(0.2)
        # # Debug: Plot the speed
        # plt.plot(speed_list)
        # plt.show()

    def execute_ee_waypoints(
        self,
        waypoints: list[list[float]],
        d_t: float = 0.008,
        to_rotvec: bool = False,
        **kwargs,
    ):
        """Execute a trajectory"""
        # Convert waypoints to rotation vector pose
        if to_rotvec:
            waypoints = [self._quat_to_rotvec_pose(p) for p in waypoints]
        # speed_list = []

        # Execute each waypoint
        for waypoint in waypoints:
            start_t = self.rtde.rtde_c.initPeriod()
            self.rtde.servo_tool(waypoint, time=d_t, **kwargs)
            self.rtde.rtde_c.waitPeriod(start_t)
            # speed_list.append(self.get_ee_speed())

        # Stop servo
        self.rtde.rtde_c.servoStop()
        time.sleep(0.2)
        # # Debug: Plot the speed
        # plt.plot(speed_list)
        # plt.show()

    def move_joint(self, joint_angles: list[float], **kwargs):
        """Move the robot to a joint configuration"""
        self.rtde.move_joint(joint_angles, **kwargs)

    def move_tool(
        self, tool_pose: list[float], to_rotvec: bool = False, **kwargs
    ):
        """Move the robot to a tool pose"""
        if to_rotvec:
            tool_pose = self._quat_to_rotvec_pose(tool_pose)
        self.rtde.move_tool(tool_pose, **kwargs)

    def _quat_to_rotvec_pose(self, quat_pose: list[float]):
        """Convert a quaternion pose to a rotation vector pose"""
        # Quaternion in wxyz format, convert to xyzw format
        quat = np.roll(quat_pose[3:], -1)  # [w, x, y, z] → [x, y, z, w]
        rotvec = R.from_quat(quat).as_rotvec()
        return list(quat_pose[:3]) + list(rotvec)

    # Gripper
    def control_gripper(self, action: str):
        """Control the gripper"""
        if action == "open":
            self.gripper.open_gripper()
        elif action == "close":
            self.gripper.close_gripper()
        else:
            raise ValueError(f"Invalid action: {action}")

    # Camera
    def get_object_pose_top(self):
        """Get the pose of the object from the top camera"""
        return self.top_cam.get_object_pose()

    def get_object_pose_hand(self):
        """Get the pose of the object from the hand camera"""
        return self.hand_cam.get_object_pose()

    # Getters
    def get_ee_pose(self):
        """Get the tool pose of the robot [x, y, z, rx, ry, rz]"""
        return self.rtde.get_tool_pose()

    def get_ee_transform(self):
        """Get the tool transform of the robot"""
        x, y, z, rx, ry, rz = self.get_ee_pose()
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        transform[:3, :3] = R.from_rotvec([rx, ry, rz]).as_matrix()
        return transform

    def get_ee_speed(self):
        """Get the speed of the robot [vx, vy, vz, wx, wy, wz]"""
        return self.rtde.get_tool_speed()

    def get_q_values(self):
        """Get the joint values of the robot"""
        return self.rtde.get_joint_values()
