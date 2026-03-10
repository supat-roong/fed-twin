import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


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


def plot_generalization_gap(pipeline_type="single_twin", metrics_file=None):
    """
    Plots the generalization gap: evaluation on training env vs. neutral eval env.
    Shows if the model overfits to its training environment or generalizes well.
    """
    if not metrics_file:
        metrics_file = get_latest_metrics(pipeline_type)

    if not metrics_file:
        print(f"Error: No {pipeline_type} metrics found.")
        return

    print(f"Analyzing generalization gap in: {metrics_file}")
    df = pd.read_csv(metrics_file)

    # Filter for EVAL mode only
    eval_df = df[df["mode"] == "EVAL"].copy()

    # Separate training env eval vs global eval env
    # Training env: typically the training twin doing self-evaluation
    # Global env: eval-twin-global or similar
    train_env_eval = eval_df[
        ~eval_df["twin_id"].str.contains("global", case=False)
    ].copy()
    global_env_eval = eval_df[
        eval_df["twin_id"].str.contains("global", case=False)
    ].copy()

    # Group by round and get mean (in case there are multiple training twins)
    train_env_stats = train_env_eval.groupby("round")["reward"].mean().reset_index()
    global_env_stats = global_env_eval.groupby("round")["reward"].mean().reset_index()

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.style.use("seaborn-v0_8-muted")

    # Plot both lines
    if not train_env_stats.empty:
        plt.plot(
            train_env_stats["round"],
            train_env_stats["reward"],
            marker="o",
            linewidth=3,
            markersize=8,
            label="Eval on Training Environment",
            color="#3498db",
            alpha=0.8,
        )

    if not global_env_stats.empty:
        plt.plot(
            global_env_stats["round"],
            global_env_stats["reward"],
            marker="s",
            linewidth=3,
            markersize=8,
            label="Eval on Neutral Environment",
            color="#e74c3c",
            alpha=0.8,
        )

    # Fill area between to show gap
    if not train_env_stats.empty and not global_env_stats.empty:
        # Align rounds
        merged = pd.merge(
            train_env_stats,
            global_env_stats,
            on="round",
            suffixes=("_train", "_global"),
        )
        plt.fill_between(
            merged["round"],
            merged["reward_train"],
            merged["reward_global"],
            alpha=0.2,
            color="gray",
            label="Generalization Gap",
        )

    # Styling
    plt.title(
        "Generalization Analysis: Training vs. Neutral Environment",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11, frameon=True, shadow=True, loc="lower right")

    # Add insight box
    if not train_env_stats.empty and not global_env_stats.empty:
        final_train = train_env_stats["reward"].iloc[-1]
        final_global = global_env_stats["reward"].iloc[-1]
        gap = final_train - final_global

        textstr = f"Final Gap: {gap:.2f}\n"
        if gap > 20:
            textstr += "High overfitting to\ntraining environment"
            color = "#e74c3c"
        elif gap > 10:
            textstr += "Moderate specialization"
            color = "#f39c12"
        else:
            textstr += "Good generalization!"
            color = "#27ae60"

        props = dict(
            boxstyle="round", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2
        )
        plt.text(
            0.02,
            0.98,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
        )

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    output_path = f"plots/generalization_gap_{pipeline_type}.png"
    plt.savefig(output_path, dpi=300)
    print(f"✓ Generalization gap chart saved to: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        # helper [type] [file]
        p_type = sys.argv[1]
        m_file = sys.argv[2]
        plot_generalization_gap(p_type, m_file)
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg.endswith(".csv"):
            # helper [file] -> infer type or default
            p_type = "fed_twin" if "fed_twin" in arg else "single_twin"
            plot_generalization_gap(p_type, arg)
        else:
            # helper [type]
            plot_generalization_gap(arg)
    else:
        plot_generalization_gap()
