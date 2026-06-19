import copy

import mujoco as mj
import numpy as np

from .motion_retarget import GeneralMotionRetargeting
from .params import ROBOT_XML_DICT


class RobotBodyFrameExtractor:
    """Extract MuJoCo body poses from a robot qpos vector."""

    def __init__(self, robot_type, body_names):
        self.robot_type = robot_type
        self.xml_file = str(ROBOT_XML_DICT[robot_type])
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        self.data = mj.MjData(self.model)
        self.body_names = list(body_names)

        missing = [name for name in self.body_names if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name) < 0]
        if missing:
            raise ValueError(f"{robot_type} is missing body names required for retargeting: {missing}")

    def extract(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape[-1] != self.model.nq:
            raise ValueError(
                f"Expected qpos with {self.model.nq} values for {self.robot_type}, "
                f"got shape {qpos.shape}."
            )

        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)

        body_frames = {}
        for body_name in self.body_names:
            body_id = self.model.body(body_name).id
            body_frames[body_name] = [
                self.data.xpos[body_id].copy(),
                self.data.xquat[body_id].copy(),
            ]
        return body_frames


class RobotToRobotRetargeting:
    """Retarget qpos trajectories from one robot model to another via GMR IK."""

    def __init__(
        self,
        src_robot="unitree_g1",
        tgt_robot="pnd_adam_pro",
        solver="daqp",
        damping=5e-1,
        verbose=True,
        use_velocity_limit=False,
        project_initial_configuration=True,
    ):
        self.src_robot = src_robot
        self.tgt_robot = tgt_robot
        self.retargeter = GeneralMotionRetargeting(
            src_human=src_robot,
            tgt_robot=tgt_robot,
            solver=solver,
            damping=damping,
            verbose=verbose,
            use_velocity_limit=use_velocity_limit,
        )
        if project_initial_configuration:
            self.project_target_configuration_to_limits()
        self.source_frame_extractor = RobotBodyFrameExtractor(
            src_robot,
            self.retargeter.human_scale_table.keys(),
        )

    @property
    def source_nq(self):
        return self.source_frame_extractor.model.nq

    @property
    def target_nq(self):
        return self.retargeter.model.nq

    def project_target_configuration_to_limits(self):
        """Clamp the initial target qpos to MuJoCo hinge/slide joint limits."""
        qpos = self.retargeter.configuration.q.copy()
        model = self.retargeter.model
        limited = model.jnt_limited.astype(bool)
        limited &= model.jnt_type != mj.mjtJoint.mjJNT_FREE

        for joint_id in np.where(limited)[0]:
            joint_type = model.jnt_type[joint_id]
            if joint_type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            qpos_addr = model.jnt_qposadr[joint_id]
            lower, upper = model.jnt_range[joint_id]
            qpos[qpos_addr] = np.clip(qpos[qpos_addr], lower, upper)

        self.retargeter.configuration.update(q=qpos)

    def make_source_qpos(self, root_pos, root_rot, dof_pos):
        root_pos = np.asarray(root_pos, dtype=float)
        root_rot = np.asarray(root_rot, dtype=float)
        dof_pos = np.asarray(dof_pos, dtype=float)
        qpos = np.concatenate([root_pos, root_rot, dof_pos])
        if qpos.shape[-1] != self.source_nq:
            raise ValueError(
                f"Expected source qpos length {self.source_nq} for {self.src_robot}, "
                f"got {qpos.shape[-1]}."
            )
        return qpos

    def extract_source_frames(self, source_qpos):
        return self.source_frame_extractor.extract(source_qpos)

    def retarget_qpos(self, source_qpos, return_ik_targets=False):
        source_frames = self.extract_source_frames(source_qpos)
        target_qpos = self.retargeter.retarget(source_frames)
        if not return_ik_targets:
            return target_qpos
        return target_qpos, copy.deepcopy(self.retargeter.scaled_human_data)

    def retarget_frame(self, root_pos, root_rot, dof_pos, return_ik_targets=False):
        source_qpos = self.make_source_qpos(root_pos, root_rot, dof_pos)
        return self.retarget_qpos(source_qpos, return_ik_targets=return_ik_targets)
