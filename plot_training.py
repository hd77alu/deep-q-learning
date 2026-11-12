"""Plot the best run's logged reward, episode length and cumulative steps."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('metrics', type=Path)
parser.add_argument('--output', type=Path, default=Path('training_plot.png'))
args = parser.parse_args()
with args.metrics.open() as file:
    rows = list(csv.DictReader(file))
if not rows:
    parser.error('No completed episodes in metrics file')
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
episodes = [int(row['Episode']) for row in rows]
for axis, key, label in zip(axes, ['Episode_Reward', 'Episode_Length', 'Total_Steps'], ['Raw reward', 'Episode length (agent steps)', 'Cumulative agent steps']):
    axis.plot(episodes, [float(row[key]) for row in rows], linewidth=1)
    axis.set_ylabel(label)
    axis.grid(alpha=.2)
axes[-1].set_xlabel('Completed episode')
fig.suptitle(f'Pac-Man DQN training: {args.metrics.parent}')
fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=160)
print(args.output)
