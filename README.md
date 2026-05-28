# TurtleBot3 SAC Navigation

Soft Actor-Critic (SAC) navigation for TurtleBot3 in a custom Gazebo environment.

This repository was extracted from a modified `ROBOTIS-GIT/turtlebot3_simulations` workspace and starts with a fresh Git history. It keeps only the custom SAC package and the minimal Gazebo assets needed for the custom environment.

This work began as a Sungkyunkwan University Undergraduate Research Program (URP) project in the summer of 2024 and received an award at the 2024 Undergraduate Academic Conference of the School of Mechanical Engineering, Sungkyunkwan University.

Korean documentation is available at [docs/README_KO.md](docs/README_KO.md).

## Results

Execution examples:

![Execution result 1](docs/assets/execution-result-1.gif)

![Execution result 2](docs/assets/execution-result-2.gif)

Training reward:

![Training reward](docs/assets/training-reward.png)

## Contents

- `sac`: ROS package `turtlebot3_sac`, containing the SAC agent, replay buffer, neural networks, ROS training nodes, validation nodes, and environment code.
- `gazebo`: ROS package `turtlebot3_gazebo`, containing the custom world, custom model, goal-box model, and launch file.
- `pretrained`: tracked pretrained checkpoint files for quick validation runs.
- `docs/assets`: README media files.
- `LICENSE.ROBOTIS`: upstream ROBOTIS license text kept for attribution and dependency clarity.

## Dependencies

This repository is not a standalone TurtleBot3 distribution. It expects a ROS Noetic catkin workspace with TurtleBot3 dependencies installed or cloned.

Required ROS packages include:

- `gazebo_ros`
- `xacro`
- `turtlebot3_description`
- `turtlebot3_msgs`
- `geometry_msgs`
- `sensor_msgs`
- `nav_msgs`
- `std_msgs`
- `std_srvs`
- `tf`
- `gazebo_msgs`

The Gazebo launch file spawns the robot through `turtlebot3_description`. If that package is missing, launch fails with:

```text
Resource not found: turtlebot3_description
```

Install the TurtleBot3 dependencies with apt:

```bash
sudo apt install ros-noetic-turtlebot3-description ros-noetic-turtlebot3-msgs
```

or clone the TurtleBot3 dependency repositories:

```bash
cd ~/catkin_ws/src
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
```

Do not clone the full upstream `ROBOTIS-GIT/turtlebot3_simulations` repository into the same catkin workspace as this repository. This repository already provides a minimal package named `turtlebot3_gazebo`, so a second package with the same name will cause package-resolution conflicts.

Python dependencies include:

- `numpy`
- `torch`
- `scipy`
- `matplotlib`
- `tensorboard` or `tensorboardX`, depending on the node used

## Build

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

If ROS Noetic accidentally uses a conda Python environment, build with the system Python:

```bash
PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages \
CMAKE_PREFIX_PATH=/opt/ros/noetic \
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

## Usage

Launch the custom Gazebo world:

```bash
roslaunch turtlebot3_gazebo basement3_world.launch
```

Launch SAC training:

```bash
roslaunch turtlebot3_sac turtlebot3_sac_stage_4.launch
```

Launch validation:

```bash
roslaunch turtlebot3_sac turtlebot3_sac_stage_1_validate.launch
```

## End-to-End Workflow

### 1. Clone and build

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/kyuhunsim/turtlebot3-sac-navigation.git
cd ~/catkin_ws
catkin_make
source /opt/ros/noetic/setup.bash
source devel/setup.bash
export TURTLEBOT3_MODEL=burger
```

If your shell is `zsh`, source `setup.zsh` instead of `setup.bash`.

If `catkin_make` picks up a conda Python and fails, rebuild with the system Python:

```bash
PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages \
CMAKE_PREFIX_PATH=/opt/ros/noetic \
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

### 2. Launch Gazebo

Terminal 1:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo basement3_world.launch
```

If Gazebo fails with `Resource not found: turtlebot3_description`, install the TurtleBot3 dependencies listed above.

If Gazebo fails with `No module named 'rospkg'`, install:

```bash
sudo apt install python3-rospkg
```

### 3. Start fresh training

Terminal 2:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_sac turtlebot3_sac_stage_4.launch
```

This uses the default fresh-training settings in `sac/node/default.py`:

```python
load_model = False
load_episode = 2500
world = 'stage_4'
```

Training outputs are written locally to:

```text
sac/SAC_model/
sac/runs/
sac/csv/
```

### 4. Load the included pretrained model and validate it

This repository includes one tracked pretrained checkpoint pair:

```text
pretrained/stage_4/2500_policy_net.pth
pretrained/stage_4/2500value_net.pth
```

Copy the files into the runtime checkpoint directory:

```bash
cd ~/catkin_ws/src/turtlebot3-sac-navigation
mkdir -p sac/SAC_model/stage_4
cp pretrained/stage_4/2500_policy_net.pth sac/SAC_model/stage_4/
cp pretrained/stage_4/2500value_net.pth sac/SAC_model/stage_4/
```

Then edit `sac/node/default.py` to load the checkpoint:

```python
load_model = True
load_episode = 2500
world = 'stage_4'
```

With Gazebo still running in Terminal 1, start validation in Terminal 2:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_sac turtlebot3_sac_stage_1_validate.launch
```

That launch file runs `validate_3.py` on the complex custom map with `/stage_number=4`.

After validation, switch `load_model` back to `False` before starting a new training run.

## Stages

In this repository, a `stage` is an experiment preset used by the SAC launch files. It does not automatically switch the Gazebo world by itself. The Gazebo world is launched separately with `turtlebot3_gazebo basement3_world.launch`, and the SAC launch file selects the training or validation behavior.

| Stage | Launch file | Purpose | Behavior |
| --- | --- | --- | --- |
| Stage 1 | `sac/launch/turtlebot3_sac_stage_1.launch` | Basic SAC training preset | Runs `train_2.py` with `/stage_number=1`. This is the simpler training setup. |
| Stage 4 | `sac/launch/turtlebot3_sac_stage_4.launch` | Main custom-map training preset | Runs `train_2.py` with `/stage_number=4` and starts `combination_obstacle_1.py` and `combination_obstacle_2.py` for moving obstacle behavior. |
| Stage 4 validation | `sac/launch/turtlebot3_sac_stage_1_validate.launch` | Validation on the complex custom map | Runs `validate_3.py` with moving obstacle scripts. Despite the filename, the launch file sets `/stage_number=4`. |
| Stage 5 validation | `sac/launch/turtlebot3_sac_stage_5_validate.launch` | Additional validation preset | Runs `validate_4.py` with `combination_obstacle_3.py` and `combination_obstacle_4.py`. |

The project result is that a model trained from a simpler navigation setup was able to generalize to the more complex custom map and avoid obstacles in the recorded runs.

## Configuration

Main training settings are in:

```text
sac/node/default.py
```

Important fields:

- `gamma`, `tau`, `lr`, `alpha`: SAC optimization parameters.
- `batch_size`, `hidden_dim`, `replay_size`: training capacity and network size.
- `max_steps`, `max_episodes`: episode length and training duration.
- `state_dim`, `action_dim`: observation and action dimensions. The current `state_dim=30` is composed of 24 downsampled laser sectors, 2 previous action values, and 4 navigation/obstacle features.
- `ACTION_V_MIN`, `ACTION_V_MAX`: linear velocity range.
- `ACTION_W_MIN`, `ACTION_W_MAX`: angular velocity range.
- `world`: checkpoint/log subdirectory name.
- `load_model`, `load_episode`: whether to resume from an existing checkpoint. Keep `load_model = False` for fresh training from a new clone.
- `/stage_number`: ROS parameter set by the launch files. It is consumed by the goal/validation logic to decide stage-specific behavior.

Training entrypoint:

```text
sac/node/train_2.py
```

Model save/load logic:

```text
sac/node/sac.py
```

Environment and reward logic:

```text
sac/node/environment_stage_1.py
sac/src/env/respawnGoal_3.py
```

## Generated Files

Running training or validation creates local experiment artifacts. These files are intentionally ignored by Git, except for the tracked release checkpoint under `pretrained/`:

- `sac/SAC_model/`: saved policy and critic checkpoints.
- `sac/runs/`: TensorBoard event logs.
- `sac/csv/`: validation CSV outputs.
- `*.pth`, `*.pt`: model weights generated during experiments. The included `pretrained/stage_4/*.pth` files are explicitly tracked.
- `events.out.tfevents*`: TensorBoard events.
- `*.ipynb`: local analysis notebooks.
- `__pycache__/`, `*.pyc`: Python cache files.

Use this to confirm ignored artifacts:

```bash
git status --ignored
```

## Validation Status

Checked during repository cleanup:

- Python syntax check passed for all `turtlebot3_sac` Python files.
- XML validation passed for package and launch files.
- `rospack find turtlebot3_sac` and `rospack find turtlebot3_gazebo` resolve to this repository.
- `roslaunch --files turtlebot3_sac turtlebot3_sac_stage_4.launch` resolves successfully.
- `catkin_make` passed in a clean temporary workspace containing only this repository and `/opt/ros/noetic`.
- `roslaunch --files turtlebot3_gazebo basement3_world.launch` requires `turtlebot3_description`; install or clone the dependency above before launching Gazebo.

## Notes

Use `train_2.py` for training and `validate_3.py` or `validate_4.py` through the launch files above for validation.
