# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import pickle as pkl
import torch
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.utils import constants
from fairmotion.data import amass

"""
Structure of pkl file in AIST dataset is as follows.
- smpl_trans (num_frames, 3):  translation (x, y, z) of root joint
- smpl_scaling (1,):
- smpl_loss (1,)
- smpl_poses (num_frames, 72): 24 3D axis-angl  joints,
    corresponding to the names in amass_dip.SMPL_JOINTS
"""

def load(
    file,
    sequence=None,
    motion=None,
    scale=1.0,
    load_skel=True,
    load_motion=True,
    v_up_skel=np.array([0.0, 1.0, 0.0]),
    v_face_skel=np.array([0.0, 0.0, 1.0]),
    v_up_env=np.array([0.0, 1.0, 0.0]),
):
    from fairmotion.data.amass_dip import OFFSETS, SMPL_JOINTS, SMPL_PARENTS
    if motion is None:
        motion = motion_class.Motion(fps=60)

    if load_skel:
        skel = motion_class.Skeleton(
            v_up=v_up_skel, v_face=v_face_skel, v_up_env=v_up_env,
        )
        smpl_offsets = np.zeros([24, 3])
        smpl_offsets[0] = OFFSETS[0]
        for idx, pid in enumerate(SMPL_PARENTS[1:]):
            smpl_offsets[idx + 1] = OFFSETS[idx + 1] - OFFSETS[pid]
        for joint_name, parent_joint, offset in zip(
            SMPL_JOINTS, SMPL_PARENTS, smpl_offsets
        ):
            joint = motion_class.Joint(name=joint_name)
            if parent_joint == -1:
                parent_joint_name = None
                joint.info["dof"] = 6  # root joint is free
                offset -= offset
            else:
                parent_joint_name = SMPL_JOINTS[parent_joint]
            offset = offset / np.linalg.norm(smpl_offsets[4])
            T1 = conversions.p2T(scale * offset)
            joint.xform_from_parent_joint = T1
            skel.add_joint(joint, parent_joint_name)
        motion.skel = skel
    else:
        assert motion.skel is not None

    if load_motion:
        assert motion.skel is not None
        # Assume 60fps
        motion.set_fps(60.0)
        dt = float(1 / motion.fps)
        # load poses from sequence
        if sequence is not None:
            sequence = sequence.reshape(-1, 72)
            poses = sequence
            assert len(poses) > 0, "sequence is empty"
        # load poses from file
        else:
            with open(file, "rb") as f:
                data = pkl.load(f)
                poses = np.array(data["smpl_poses"])
                trans = np.array(data["smpl_trans"])
                trans /= 100 # m to cm
                assert len(poses) > 0, "file is empty"

        poses = poses.reshape((-1, len(SMPL_JOINTS), 3))

        for pose_id, pose in enumerate(poses):
            trans_data = trans[pose_id]
            pose_data = [
                constants.eye_T() for _ in range(len(SMPL_JOINTS))
            ] # identity transition matrices
            for joint_id, joint_name in enumerate(SMPL_JOINTS):
                if joint_id == 0:
                    pose_data[
                        motion.skel.get_index_joint(joint_name)
                    ] = conversions.Rp2T(
                            conversions.A2R(pose[joint_id]), trans_data
                        )
                else:
                    pose_data[
                        motion.skel.get_index_joint(joint_name)
                    ] = conversions.A2T(pose[joint_id])
            motion.add_one_frame(pose_data)

    return motion

def _load(file, bm=None, bm_path=None, model_type="smplh"):
    from human_body_prior.body_model.body_model import BodyModel
    num_betas = 10
    if bm is None:
        # Download the required body model. For SMPL-H download it from
        # http://mano.is.tue.mpg.de/.
        assert bm_path is not None, "Please provide SMPL body model path"
        from pathlib import Path
        import tempfile
        import scipy.sparse
        bm_path = Path(bm_path).resolve()
        assert bm_path.exists(), "Please provide valid SMPL body model path"
        with tempfile.TemporaryDirectory() as tmpdirname:
            # this is a hack to make it possible to load from pkl
            if bm_path.suffix == ".pkl":
                hack_bm_path = Path(tmpdirname) / (bm_path.stem + ".npz")
                with open(bm_path, "rb") as f:
                    try:
                        data = pkl.load(f, encoding="latin1")
                    except ModuleNotFoundError as e:
                        if "chumpy" in str(e):
                            message = ("Failed to load pickle file because "
                                "chumpy is not installed.\n"
                                "The original SMPL body model archives store "
                                "some arrays as chumpy arrays, these are cast "
                                "back to numpy arrays before use but it is not "
                                "possible to unpickle the data without chumpy "
                                "installed.")
                            raise ModuleNotFoundError(message) from e
                        else:
                            raise e

                    def clean(x):
                        if 'chumpy' in str(type(x)):
                            return np.array(x)
                        elif type(x) == scipy.sparse.csc.csc_matrix:
                            return x.toarray()
                        else:
                            return x

                    data = {k: clean(v) for k, v in data.items() if type(v)}
                    data = {k: v for k, v in data.items() if type(v) == np.ndarray}
                    np.savez(hack_bm_path, **data)
            else:
                hack_bm_path = bm_path
            bm = amass.load_body_model(str(hack_bm_path), num_betas, model_type)

    skel = amass.create_skeleton_from_amass_bodymodel(
        bm, None, len(amass.joint_names), amass.joint_names,
    )

    with open(file, "rb") as f:
        bdata = pkl.load(f)
        fps = 60
        root_orient = bdata["smpl_poses"][:, :3]  # controls the global root orientation
        pose_body = bdata["smpl_poses"][:, 3:66]  # controls body joint angles
        trans = bdata["smpl_trans"][:, :3] / bdata["smpl_scaling"][0]  # controls global position

        motion = motion_class.Motion(skel=skel, fps=fps)

        num_joints = skel.num_joints()
        parents = bm.kintree_table[0].long()[:num_joints]

        for frame in range(pose_body.shape[0]):
            pose_body_frame = pose_body[frame]
            root_orient_frame = root_orient[frame]
            root_trans_frame = trans[frame]
            pose_data = []
            for j in range(num_joints):
                if j == 0:
                    T = conversions.Rp2T(
                        conversions.A2R(root_orient_frame), root_trans_frame
                    )
                else:
                    T = conversions.R2T(
                        conversions.A2R(
                            pose_body_frame[(j - 1) * 3: (j - 1) * 3 + 3]
                        )
                    )
                pose_data.append(T)
            motion.add_one_frame(pose_data)

        grounded_motion = motion_ops.fix_height(motion, axis_up="y")
        return grounded_motion

if __name__ == "__main__":
    motion = load("../../tests/data/aistplusplus_sample.pkl")
    _motion = _load("../../tests/data/aistplusplus_sample.pkl", bm_path="../../tests/body_models/model.npz")
    print(motion)
    print(_motion)
    motion = np.load("../../tests/data/aistplusplus_sample.pkl", allow_pickle=True)
    motion = load(None, sequence=motion['smpl_poses'])
    print(motion)
    # motion with translation
    import bvh
    motion = load("../../tests/data/aistplusplus_sample.pkl")
    print(motion)
    bvh.save(motion, "output.bvh")