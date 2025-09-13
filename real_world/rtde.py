import rtde_control
import rtde_receive

from typing import List


class RTDE:
    def __init__(self, robot_ip: str = "192.168.0.100"):
        """Initialize with the robot IP address"""
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    # Getters
    def get_joint_values(self) -> List[float]:
        """Get the joint positions in radians."""
        return self.rtde_r.getActualQ()

    def get_joint_speed(self) -> List[float]:
        """Get the joint speed in radians per second."""
        return self.rtde_r.getActualQd()

    def get_tool_pose(self) -> List[float]:
        """
        Get the pose of the Tool Center Point (TCP) in Cartesian space.

        Return [x, y, z, rx, ry, rz] position + rotation vector
        """
        return self.rtde_r.getActualTCPPose()

    def get_tool_speed(self) -> List[float]:
        """
        Get the speed of the Tool Center Point (TCP) in Cartesian space.

        Return [vx, vy, vz, wx, wy, wz]
        """
        return self.rtde_r.getActualTCPSpeed()

    def set_tool_pose(self, tcp: List[float]):
        """
        Set the Tool Center Point (TCP) in Cartesian space.

        The pose is defined as [x, y, z, rx, ry, rz] position + rotation vector
        """
        self.rtde_c.setTcp(tcp)

    # Joint control
    def move_joint(
        self,
        joint_values: List[float],
        speed: float = 1.05,
        acceleration: float = 1.4,
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """Move the robot to the target joint positions."""
        self.rtde_c.moveJ(joint_values, speed, acceleration, asynchronous)

    def move_joint_trajectory(
        self,
        path: List[List[float]],
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """
        Move the robot to follow a given path/trajectory,
        with each waypoint defined as
        [q1, q2, q3, q4, q5, q6, speed, acceleration, blend]
        (angles + others)
        """
        self.rtde_c.moveJ(path, asynchronous)

    def speed_joint(
        self,
        speeds: List[float],
        acceleration: float = 0.5,
        time: float = 0.0,
    ):
        """Accelerate linearly and continue with constant joint speed."""
        self.rtde_c.speedJ(speeds, acceleration, time)

    # Tool control
    def move_tool(
        self,
        pose: List[float],
        speed: float = 0.25,
        acceleration: float = 1.2,
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """Move the robot to the tool position."""
        self.rtde_c.moveL(pose, speed, acceleration, asynchronous)

    def move_tool_trajectory(
        self,
        path: List[List[float]],
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """
        Move the robot to follow a given tool path/trajectory,
        with each waypoint defined as
        [x, y, z, rx, ry, rz, speed, acceleration, blend]
        (position + rotation vector + others)
        """
        self.rtde_c.moveL(path, asynchronous)

    def speed_tool(
        self,
        speeds: List[float],
        acceleration: float = 0.25,
        time: float = 0.0,
    ):
        """Accelerate linearly and continue with constant joint speed."""
        self.rtde_c.speedL(speeds, acceleration, time)

    # Servo control
    def servo_joint(
        self,
        joint_values: List[float],  # Target joint positions
        speed: float = 0,  # joint velocity
        acceleration: float = 0,  # joint acceleration
        time: float = 0.008,  # time to control the robot
        lookahead_time: float = 0.1,  # project the current position forward
        gain: int = 300,  # P-term as in PID controller
    ):
        """
        Used to perform online realtime joint control

        The gain parameter works the same way as the P-term of a PID controller
        where it adjusts the current position towards the desired (q).
        The higher the gain, the faster reaction the robot will have.

        The parameter lookahead_time is used to project the current position
        forward in time with the current velocity. A low value gives fast
        reaction, a high value prevents overshoot.

        Note: A high gain or a short lookahead time may cause instability and
        vibrations. Especially if the target positions are noisy or updated
        at a low frequency It is preferred to call this function
        with a new setpoint (q) in each time step
        """
        self.rtde_c.servoJ(
            joint_values, speed, acceleration, time, lookahead_time, gain
        )

    def servo_tool(
        self,
        pose: List[float],  # Target joint positions
        speed: float = 0,  # joint velocity
        acceleration: float = 0,  # joint acceleration
        time: float = 0.008,  # time to control the robot
        lookahead_time: float = 0.1,  # project the current position forward
        gain: int = 300,  # P-term as in PID controller
    ):
        """
        Used to perform online realtime tool control

        The gain parameter works the same way as the P-term of a PID controller
        where it adjusts the current position towards the desired (q).
        The higher the gain, the faster reaction the robot will have.

        The parameter lookahead_time is used to project the current position
        forward in time with the current velocity. A low value gives fast
        reaction, a high value prevents overshoot.

        Note: A high gain or a short lookahead time may cause instability and
        vibrations. Especially if the target positions are noisy or updated
        at a low frequency It is preferred to call this function
        with a new setpoint (q) in each time step
        """
        # Tool pose [x, y, z, rx, ry, rz] position + rotation vector
        self.rtde_c.servoL(
            pose, speed, acceleration, time, lookahead_time, gain
        )

    # Stop
    def stop(
        self,
        a: float = 2.0,  # joint acceleration
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """Stop the robot."""
        self.rtde_c.stopJ(a, asynchronous)

    def stop_script(self):
        """Terminate the script on controller"""
        self.rtde_c.stopScript()


if __name__ == "__main__":
    # Test the RTDE class
    rtde = RTDE()
    print("Joint values:", rtde.get_joint_values())
    print("Joint speed:", rtde.get_joint_speed())
    print("Tool pose:", rtde.get_tool_pose())
    print("Tool speed:", rtde.get_tool_speed())
    rtde.stop_script()
