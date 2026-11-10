import argparse
import pandas as pd
from pathlib import Path


def summarize_member_experiments(member_path):
    member_dir = Path(member_path)
    files = sorted(member_dir.glob("exp_*/metrics.csv"))
    summary = []
    for filepath in files:
        df = pd.read_csv(filepath)
        exp_id = filepath.parent.name
        avg_reward = df["Episode_Reward"].tail(20).mean() # last 20 episodes
        max_reward = df["Episode_Reward"].max()
        summary.append({"Experiment": exp_id, "Avg_Final_Reward": avg_reward, "Max_Reward": max_reward})
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize member experiment results")
    parser.add_argument(
        "member_path",
        nargs="?",
        help="Member experiment folder, such as hamed",
    )
    args = parser.parse_args()

    member_path = args.member_path or input(
        "Enter the experiment folder path (for example, hamed): "
    ).strip()
    if not member_path:
        parser.error("An experiment folder path is required.")

    results = summarize_member_experiments(member_path)
    if results.empty:
        print(f"No experiment metrics found in {member_path!r}.")
    else:
        print(results.to_string(index=False))