# TurtleBot3 SAC Basement3

This repository contains a custom Soft Actor-Critic (SAC) training setup for TurtleBot3 navigation in a custom Gazebo `basement3` environment.

The code was extracted from a modified `ROBOTIS-GIT/turtlebot3_simulations` workspace and intentionally starts with a fresh Git history.

## Contents

- `turtlebot3_sac`: SAC agent, replay buffer, neural networks, ROS training nodes, validation nodes, and environment code.
- `turtlebot3_gazebo`: minimal custom Gazebo package containing the `basement3` model, goal-box model, `basement3.world`, and a launch file for the environment.
- `LICENSE.ROBOTIS`: license text from the upstream ROBOTIS TurtleBot3 simulations repository. Keep this file because this project still uses TurtleBot3/Gazebo integration patterns and may depend on upstream TurtleBot3 packages.

## Local Artifacts

Training outputs are kept locally but ignored by Git:

- `turtlebot3_sac/SAC_model/`
- `turtlebot3_sac/runs/`
- `turtlebot3_sac/csv/`
- `*.pth`, `*.pt`, TensorBoard event files, notebooks, and Python cache files

Use `git status --ignored` to confirm they are ignored.

## Dependencies

This repository is not a standalone TurtleBot3 distribution. It expects a ROS Noetic catkin workspace with TurtleBot3 dependencies installed or cloned.

Required ROS packages include:

- `gazebo_ros`
- `xacro`
- `turtlebot3_description`
- `turtlebot3_msgs`
- standard ROS message packages such as `geometry_msgs`, `sensor_msgs`, `nav_msgs`, `std_msgs`, and `std_srvs`

The launch file spawns a TurtleBot3 model using `turtlebot3_description`. If that package is missing, Gazebo launch will fail with `Resource not found: turtlebot3_description`.

Do not clone the full upstream `ROBOTIS-GIT/turtlebot3_simulations` package into the same `src` folder together with this repository, because this repository already provides a minimal package named `turtlebot3_gazebo`. Having two packages with the same name in one catkin workspace causes package-resolution conflicts.

Use one of these dependency approaches:

```bash
sudo apt install ros-noetic-turtlebot3-description ros-noetic-turtlebot3-msgs
```

or clone the upstream TurtleBot3 dependency repository, not the simulations repository:

```bash
cd ~/catkin_ws/src
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
```

Python dependencies include:

- `numpy`
- `torch`
- `scipy`
- `matplotlib`
- `tensorboard` or `tensorboardX`, depending on the node used

## Basic Usage

Set the TurtleBot3 model:

```bash
export TURTLEBOT3_MODEL=burger
```

Build from the catkin workspace root:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Launch the custom Gazebo world:

```bash
roslaunch turtlebot3_gazebo basement3_world.launch
```

Launch SAC training:

```bash
roslaunch turtlebot3_sac turtlebot3_sac_stage_4.launch
```

## Notes

Some older scripts are still experiment-specific variants. Before publishing a polished release, choose the official train/validation entrypoints and remove unused copies.

## Validation Status

Checked during repository cleanup:

- Python syntax check passed for all `turtlebot3_sac` Python files.
- XML validation passed for package and launch files.
- `rospack find turtlebot3_sac` and `rospack find turtlebot3_gazebo` resolve to this repository.
- `roslaunch --files turtlebot3_sac turtlebot3_sac_stage_4.launch` resolves successfully.
- `catkin_make` passed in a clean temporary workspace containing only this repository and `/opt/ros/noetic`.
- `roslaunch --files turtlebot3_gazebo basement3_world.launch` currently requires `turtlebot3_description`; install or clone the dependency above before launching Gazebo.
