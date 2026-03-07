import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


def get_latest_metrics(pipeline_type):
    """Gets the latest metrics file for a given pipeline type (visual or regular)."""
    # Check both visual and regular patterns
    patterns = [
        f"metrics_{pipeline_type}_visual_*.csv",
        f"metrics_{pipeline_type}_*.csv",
    ]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(f"metrics/{pattern}"))

    if not all_files:
        return None
    return max(all_files, key=os.path.getmtime)


def plot_worker_diversity(fl_file=None):
    # We focus on the FL run because it has multiple workers
    if not fl_file:
        fl_file = get_latest_metrics("fl")

    if not fl_file:
        print("Error: No Federated Learning metrics found.")
        return

    print(f"Analyzing Worker Diversity in: {fl_file}")
    df = pd.read_csv(fl_file)

    # Filter data
    train_df = df[df["mode"] == "TRAIN"].copy()
    eval_df = df[df["mode"] == "EVAL"].copy()

    # Identify the global eval twin vs local eval results
    global_eval = eval_df[eval_df["twin_id"].str.contains("global", case=False)].copy()

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    # 1. Plot local training rewards (faded background)
    twins = train_df["twin_id"].unique()
    colors = sns.color_palette("husl", len(twins))

    for i, twin in enumerate(twins):
        twin_data = train_df[train_df["twin_id"] == twin].sort_values("round")
        plt.plot(
            twin_data["round"],
            twin_data["reward"],
            marker="",
            color=colors[i],
            alpha=0.3,
            linewidth=1,
            linestyle="--",
            label="_nolegend_",
        )

    # 2. Plot Training Mean with Shaded Variance (Standard Deviation)
    train_stats = train_df.groupby("round")["reward"].agg(["mean", "std"]).reset_index()
    plt.fill_between(
        train_stats["round"],
        train_stats["mean"] - train_stats["std"],
        train_stats["mean"] + train_stats["std"],
        color="blue",
        alpha=0.1,
        label="Worker Training Variance",
    )
    plt.plot(
        train_stats["round"],
        train_stats["mean"],
        color="blue",
        linewidth=2,
        label="Mean Local Training Reward",
    )

    # 3. Plot Global Evaluation (High Emphasis)
    if not global_eval.empty:
        global_eval = global_eval.sort_values("round")
        plt.plot(
            global_eval["round"],
            global_eval["reward"],
            color="gold",
            linewidth=4,
            marker="*",
            markersize=15,
            label="Aggregated Global Model (Eval)",
            path_effects=None,
        )  # gold stands out

        # Add a shadow/glow effect manually
        plt.plot(
            global_eval["round"],
            global_eval["reward"],
            color="black",
            linewidth=5,
            alpha=0.2,
            zorder=1,
        )

    # 4. Final Touches
    plt.title(
        "Worker Diversity & Global Convergence", fontsize=18, fontweight="bold", pad=25
    )
    plt.xlabel("Round", fontsize=13)
    plt.ylabel("Average Reward", fontsize=13)

    plt.legend(
        loc="upper left", bbox_to_anchor=(1, 1), frameon=True, shadow=True, fontsize=11
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)

    # Summary Box
    if not train_stats.empty:
        final_mean = train_stats["mean"].iloc[-1]
        final_std = train_stats["std"].iloc[-1]
        textstr = "\n".join(
            (
                f"Final Mean: {final_mean:.2f}",
                f"Final Std: {final_std:.2f}",
                f"Total Workers: {len(twins)}",
            )
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        plt.text(
            0.02,
            0.05,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=props,
        )

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    output_path = "plots/worker_diversity.png"
    plt.savefig(output_path, dpi=300)
    print(f"✓ Worker diversity plot saved to: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        fl_path = sys.argv[1]
        plot_worker_diversity(fl_path)
    else:
        plot_worker_diversity()
