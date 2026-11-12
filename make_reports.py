"""Generate assignment tables from saved run configurations and episode logs."""
import csv
import json
from pathlib import Path

HAMED = [
(1e-4,.99,32,1,.01,50000),(2.5e-4,.99,32,1,.01,50000),(5e-5,.99,32,1,.01,50000),
(1e-4,.99,64,1,.01,50000),(1e-4,.99,16,1,.01,50000),(1e-4,.95,32,1,.01,50000),
(1e-4,.995,32,1,.01,50000),(1e-4,.99,32,1,.01,70000),(1e-4,.99,32,1,.01,25000),
(1e-4,.99,32,1,.05,50000)]

def make_report(member):
    directory = Path(member)
    rows = []
    for index in range(1,11):
        folder = directory / f'exp_{index:02d}'
        metrics_path = folder / 'metrics.csv'
        if not metrics_path.exists():
            continue
        with metrics_path.open() as file:
            metrics = list(csv.DictReader(file))
        if not metrics:
            continue
        config_path = folder / 'config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            values = tuple(config[k] for k in ('lr','gamma','batch_size','eps_start','eps_end','eps_decay'))
        elif member == 'hamed':
            values = HAMED[index-1]
        else:
            raise ValueError(f'Missing configuration: {config_path}')
        rewards = [float(row['Episode_Reward']) for row in metrics]
        first = sum(rewards[:20])/min(20,len(rewards))
        last = sum(rewards[-20:])/min(20,len(rewards))
        trend = 'rose' if last > first else 'fell'
        observation = f'First 20 mean {first:.1f}; last 20 mean {last:.1f} ({trend}); peak {max(rewards):.0f}; {len(metrics)} completed episodes. Last logged step {metrics[-1]["Total_Steps"]}.'
        rows.append(f'| {index} | '+ ' | '.join(str(v) for v in values) + f' | {observation} |')
    name = {'hamed':'Hamed','andrew':'Andrew','mahamat_bakhit':'MahamatBakhit'}.get(member, member.title())
    display_name = 'Mahamat Bakhit' if member == 'mahamat_bakhit' else name
    output = directory / f'{name}Hyperparameter_experiment_results.md'
    output.write_text(f'# Hyperparameter experiment results — {display_name}\n\nEnvironment: `ALE/Pacman-v5`. Rewards are raw training episode rewards under epsilon-greedy actions, not greedy evaluation scores. Epsilon decay is measured in agent steps.\n\n| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |\n|---|---:|---:|---:|---:|---:|---:|---|\n'+'\n'.join(rows)+'\n')
    print(output)

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('members',nargs='+')
    for member in parser.parse_args().members:
        make_report(member)
