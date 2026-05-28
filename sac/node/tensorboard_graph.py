#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Use Agg backend for headless environments
plt.switch_backend('Agg')

# TensorBoard log directory.
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
log_dir = os.path.join(package_dir, 'runs', 'stage4')

# Get all files in the directory.
event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]

# Configure the plot.
plt.figure(figsize=(10, 6))

# Process each event file.
for event_file in event_files:
    # Load TensorBoard log data with EventAccumulator.
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Extract scalar data, for example the 'reward/train' scalar.
    try:
        scalars = ea.scalars.Items('reward/train')

        # Split scalar events into step and value lists.
        steps = [scalar.step for scalar in scalars]
        values = [scalar.value for scalar in scalars]

        # Plot the reward curve.
        plt.plot(steps, values, label=os.path.basename(event_file))
    except KeyError:
        print(f'Key "reward/train" not found in {event_file}')

# Configure plot labels and title.
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Training reward Over episodes')

# Add a legend only when labels are available.
if plt.gca().has_data():
    plt.legend()

plt.grid(True)

# Save the plot as a PNG file
plt.savefig('training_reward_plot.png')
print('Plot saved as training_reward_plot.png')
