# lfg

Learnable factor graph models for ball trajectory estimation and rollout, built around PyTorch and Hydra. The repository contains training code, pretrained-weight hooks, real and synthetic datasets, and a ROS1 inference node for tennis-ball tracking.


## Installation

```bash
pip3 install -e .
```

If you plan to use the ROS node, you also need a ROS1 environment with the required message packages available at runtime.

## Pretrained weights

Some inference code loads weights from a root-level `logdir/` folder. In particular, the ROS rollout path in [lfg/derive_mnnl.py](/home/qxiao33/lfg/lfg/derive_mnnl.py) expects weights under:

```text
logdir/traj_train/MNNL/pos/real_tennis/OptimLayer/run00/
```

Download the pretrained `logdir` archive here:

[Download pretrained `logdir` archive](https://u.pcloud.link/publink/show?code=kZYP255ZjH4GtDoNoIzFNnD502pe5SzagVm7)

Place the extracted `logdir/` directory at the project root so the layout looks like:

```text
lfg/
├── conf/
├── data/
├── draw_util/
├── lfg/
├── logdir/
├── tests/
├── LICENSE
├── README.md
└── setup.py
```




## ROS1 rollout demo

The ROS node is [lfg/lfg_node_reset.py](/home/qxiao33/lfg/lfg/lfg_node_reset.py). It is not packaged as a full catkin package here, so the intended flow is to copy the script into an existing ROS1 package and run it there.

1. Copy `lfg/lfg_node_reset.py` into `~/catkin_ws/src/<your_package>/scripts/`
2. Make it executable with `chmod +x`
3. Start `roscore`
4. Run the node with `rosrun <your_package> lfg_node_reset.py`
5. Play a compatible bag file
6. Visualize the published topics in `rviz`

The node publishes (frame_id = "world"):

- `/ball/rollout/path` as `nav_msgs/Path`
- `/ball/rollout/pos` as `geometry_msgs/PoseStamped`
- `/ball/rollout/is_bounce` as `std_msgs/Bool`
- `/tennis_court_markers` as `visualization_msgs/Marker`

The current node subscribes to:

- `/camera_1/detector_1/detections`
- `/camera_2/detector_2/detections`
- `/camera_3/detector_3/detections`
- `/wcodometry_global` (for filtering out noise around robot)


## Test bag files

Bag files for ROS testing can be downloaded here:

[Download ROS test bag files](https://api.pcloud.com/getpubzip?code=kZAHi45ZtPw8foMlIoYymMXVv5rpmyWBaaHV)

## Notes

- The repo mixes research code, experiments, and runnable utilities. Not everything under `tests/` is a unit test.
- Some scripts assume specific pretrained weights or dataset layouts already exist.
- If you want a different model or dataset, start by editing the Hydra configs in `conf/`.
