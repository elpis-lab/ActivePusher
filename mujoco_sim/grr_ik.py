import numpy as np

from expansion_grr.bullet_api.loader import load_grr

from geometry.trajectory import SplineTrajectory


class IK:
    """
    An IK package that uses Expansion GRR that
    converts workspace path to robot joint trajectory
    """

    def __init__(self, robot: str):
        """Initialize with a robot name"""
        self.grr = load_grr(robot, "rot_variable_yaw")

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        none_on_fail: bool = False,
    ):
        """Solve IK for target point"""
        return self.grr.solve(target, current, none_on_fail=none_on_fail)

    def ws_path_to_traj(
        self,
        t_path: np.ndarray,
        ws_path: np.ndarray,
        none_on_fail: bool = False,
    ) -> SplineTrajectory | None:
        """Convert a workspace path with time stamps to a trajectory"""
        t_path = np.asarray(t_path)
        ws_path = np.asarray(ws_path)

        # Pose uses quaternion in wxyz format while GRR uses xyzw format
        ws_path = ws_path[:, [0, 1, 2, 4, 5, 6, 3]]

        # Generate Configuration Path with GRR
        # keep Giving the Last Solution as a Reference
        c_path = [None]
        for target in ws_path:
            solution = self.solve(target, c_path[-1], none_on_fail)
            if none_on_fail and solution is None:
                print(f"IK failed at solving {target}")
                print(f"You may consider setting none_on_fail to False")
                return None
            else:
                c_path.append(solution)
        c_path = np.array(c_path[1:])

        # Convert to Spline Trajectory
        trajectory = SplineTrajectory(c_path, t_path)
        return trajectory
