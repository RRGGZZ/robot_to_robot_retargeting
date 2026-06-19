# Unitree G1 to PND Adam Pro Retargeting

This path retargets saved Unitree G1 robot motions to PND Adam Pro robot motions.
It reuses the existing GMR inverse-kinematics pipeline by treating selected G1
body frames as the source "human" frames and solving Adam Pro IK targets.

## Model Note

The supplied Adam Pro URDF at `assets/pnd_models/adam_pro/adam_pro.urdf` is kept
unchanged. GMR registers `pnd_adam_pro` with
`assets/pnd_models/adam_pro/adam_pro.xml` because that MuJoCo XML loads directly
and corresponds to the same Adam Pro asset.

## Offline Conversion

```bash
python scripts/g1_to_adam_pro.py \
  --g1_motion_path path/to/g1_motion.pkl \
  --save_path path/to/adam_pro_motion.pkl
```

The input pickle may contain either:

- `qpos`: full Unitree G1 MuJoCo qpos, or
- `root_pos`, `root_rot`, and `dof_pos` in the repository motion format.

By default `root_rot` is interpreted as `xyzw`, matching the existing
`load_robot_motion()` convention. Use `--root_rot_order wxyz` for files saved in
MuJoCo scalar-first order.

## Visualize

Convert and visualize in one pass:

```bash
python scripts/g1_to_adam_pro.py \
  --g1_motion_path path/to/g1_motion.pkl \
  --save_path path/to/adam_pro_motion.pkl \
  --visualize \
  --rate_limit
```

Visualize a saved Adam Pro output:

```bash
python scripts/vis_g1_to_adam_pro.py \
  --adam_motion_path path/to/adam_pro_motion.pkl \
  --loop \
  --rate_limit
```

The Adam Pro output includes `ik_target_frames` by default, so the visualization
can overlay the G1-derived source frames used as IK targets. If those were not
saved, pass `--g1_motion_path path/to/g1_motion.pkl` to rebuild the overlay.

## Real-Time API

The retargeting implementation is not limited to pickle files. The pickle
scripts are only offline wrappers around the reusable streaming API.

Instantiate the retargeter once:

```python
from general_motion_retargeting import RobotToRobotRetargeting

retargeter = RobotToRobotRetargeting(
    src_robot="unitree_g1",
    tgt_robot="pnd_adam_pro",
    verbose=False,
)

# Optional: reduce IK iterations for online control.
retargeter.retargeter.max_iter = 5
```

Then call it once per frame:

```python
g1_qpos = retargeter.make_source_qpos(
    root_pos=g1_root_pos,
    root_rot=g1_root_quat_wxyz,
    dof_pos=g1_dof_pos,
)

adam_qpos = retargeter.retarget_qpos(g1_qpos)
adam_root_pos = adam_qpos[:3]
adam_root_quat_wxyz = adam_qpos[3:7]
adam_dof_pos = adam_qpos[7:]
```

Expected dimensions:

- `g1_root_pos`: `(3,)`
- `g1_root_quat_wxyz`: `(4,)`, MuJoCo scalar-first quaternion
- `g1_dof_pos`: `(29,)`
- `g1_qpos`: `(36,)`
- `adam_qpos`: `(62,)`
- `adam_dof_pos`: `(55,)`

If the upstream Unitree G1 source only provides joint positions, provide a fixed
or estimated floating-base pose before calling `make_source_qpos()`. The bridge
needs the full G1 qpos because it runs MuJoCo forward kinematics on the source
robot before solving Adam Pro IK.

For lower latency, keep the same `RobotToRobotRetargeting` instance alive across
frames. The underlying Adam Pro IK configuration is stateful, so each frame uses
the previous frame's solution as the initial guess.

## Direct Full-Qpos Input

If your streaming source already provides the full Unitree G1 MuJoCo qpos:

```python
adam_qpos = retargeter.retarget_qpos(g1_qpos)
```

Use `return_ik_targets=True` when you also want the G1-derived body frames for
visualization or debugging:

```python
adam_qpos, ik_targets = retargeter.retarget_qpos(
    g1_qpos,
    return_ik_targets=True,
)
```
