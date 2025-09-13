import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

from geometry.trajectory import SplineTrajectory


class IK:
    """
    An IK package that uses mink that
    converts workspace path to robot joint trajectory
    """

    def __init__(
        self,
        xml: str,
        ee_site: str = "attachment_site",
        default_q: np.ndarray | None = None,
        solver: str = "daqp",
        collision_pairs: list[tuple[list[str], list[str]]] = (),
        max_velocities: dict[str, float] = {},
    ):
        # Load MuJoCo model
        if "<mujoco" in xml:
            self.model = mujoco.MjModel.from_xml_string(xml)
        else:
            self.model = mujoco.MjModel.from_xml_path(xml)
        if default_q is None:
            default_q = np.zeros(self.model.nq)
        self.default_q = np.array(default_q)

        # Mink
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data
        # solver
        self.solver = solver

        # Tasks
        self.ee_site = ee_site
        self.ee_task = mink.FrameTask(
            frame_name=self.ee_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1e-3)
        self.tasks = [self.ee_task, self.posture_task]

        # Joint limits
        joint_limit = mink.ConfigurationLimit(model=self.model)
        self.limits = [joint_limit]

        # Collision avoidance
        if collision_pairs:
            collision = mink.CollisionAvoidanceLimit(
                model=self.model, geom_pairs=collision_pairs
            )
            self.limits.append(collision)

        # Velocity limit
        if max_velocities != {}:
            velocity_limit = mink.VelocityLimit(self.model, max_velocities)
            self.limits.append(velocity_limit)

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        none_on_fail: bool = False,
        iter_dt: float = 0.005,
        max_iters: int = 1000,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-3,
    ):
        """
        Solve IK for target point
        target: length-7 pose [x y z qw qx qy qz]
        """
        # Set current configuration
        if current is None:
            current = self.default_q.copy()
        self.configuration.update(q=current)
        self.posture_task.set_target(current)

        # Set target
        target_rot = mink.SO3(target[3:])
        self.ee_task.set_target(
            mink.SE3.from_rotation_and_translation(target_rot, target[:3])
        )

        # debug
        # viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Run IK
        reached = False
        for i in range(max_iters):
            # Run one IK step (joint velocities)
            qdot = mink.solve_ik(
                self.configuration,
                self.tasks,
                iter_dt,
                self.solver,
                limits=self.limits,
            )
            # Integrate to new configuration
            self.configuration.integrate_inplace(qdot, iter_dt)

            # Check convergence
            err = self.ee_task.compute_error(self.configuration)
            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            if pos_err < pos_tol and rot_err < rot_tol:
                reached = True
                break

        if not reached and none_on_fail:
            return None
        return self.configuration.q.copy()

    def ws_path_to_traj(
        self,
        t_path: np.ndarray,
        ws_path: np.ndarray,
        none_on_fail: bool = False,
    ) -> SplineTrajectory | None:
        """Convert a workspace path with time stamps to a trajectory"""
        t_path = np.asarray(t_path)
        ws_path = np.asarray(ws_path)

        # Generate Configuration Path
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


class UR10IK(IK):
    def __init__(self, xml: str, scene_included: bool = True):
        # Set up collision pairs and velocities limits
        robot_links = [
            "shoulder_collision",
            "upperarm_collision_0",
            "upperarm_collision_1",
            "forearm_collision_0",
            "forearm_collision_1",
            "wrist_1_collision",
            "wrist_2_collision_0",
            "wrist_2_collision_1",
        ]
        scene_links = ["lab_scene_collision_geom"]
        if scene_included:
            collision_pairs = [(robot_links, scene_links)]
        else:
            collision_pairs = []

        max_velocities = {
            "shoulder_pan_joint": np.pi,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }

        super().__init__(
            xml, collision_pairs=collision_pairs, max_velocities=max_velocities
        )

        # Set up default joint configuration
        default_q = np.zeros(self.model.nq)
        home_q = np.array([np.pi / 2, -1.7, 2, -1.87, -np.pi / 2, 0])
        default_q[:6] = home_q
        self.default_q = default_q

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        none_on_fail: bool = False,
        iter_dt: float = 0.005,
        max_iters: int = 1000,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-3,
    ):
        ref = current
        if ref is not None:
            ref = self.default_q.copy()
            ref[:6] = current
        solution = super().solve(
            target, ref, none_on_fail, iter_dt, max_iters, pos_tol, rot_tol
        )
        if solution is not None:
            solution = solution[:6]
        return solution


def test_ik():
    # Initialize IK
    xml = "assets/ur10_rod_ik.xml"
    ik = UR10IK(xml)

    # Launch Sim
    mj_model = mujoco.MjModel.from_xml_path(xml)
    mj_data = mujoco.MjData(mj_model)
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Solve and Run
    ws_path = np.array(
        [
            [-0.5, -0.7, 0.1, 0, 1, 0, 0],
            [0.5, -0.7, 0.3, 0, 1, 0, 0],
        ]
    )
    t_path = np.linspace(0, len(ws_path), len(ws_path))
    traj = ik.ws_path_to_traj(t_path, ws_path, none_on_fail=False)
    waypoints = traj.to_step_waypoints(0.02)

    for waypoint in waypoints:
        mj_data.qpos[:6] = waypoint
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        time.sleep(0.02)
    viewer.close()


if __name__ == "__main__":
    test_ik()
