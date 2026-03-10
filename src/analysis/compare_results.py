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


def plot_comparison(fl_file=None, single_file=None):
    # Get the latest FL and Single metrics (checks both visual and regular)
    if not fl_file:
        fl_file = get_latest_metrics("fed_twin")
    if not single_file:
        single_file = get_latest_metrics("single_twin")

    if not fl_file or not single_file:
        print(
            f"Error: Missing metrics files. Found Fed-Twin: {fl_file}, Single-Twin: {single_file}"
        )
        return

    print("Comparing:")
    print(f"  Fed-Twin: {fl_file}")
    print(f"  Single-Twin: {single_file}")

    # Load data
    df_fl = pd.read_csv(fl_file)
    df_single = pd.read_csv(single_file)

    # Filter for EVAL mode and global/representative twins
    # FL: Look for 'eval-twin-global' or any 'eval-twin'
    # Single: Look for 'eval-twin-global' or the main twin

    # Heuristic: We want the one that represents the global performance
    eval_fl = df_fl[
        (df_fl["mode"] == "EVAL")
        & (df_fl["twin_id"].str.contains("global", case=False))
    ].copy()
    if eval_fl.empty:
        # Fallback to average of all EVALs per round if global not found
        eval_fl = (
            df_fl[df_fl["mode"] == "EVAL"]
            .groupby("round")["reward"]
            .mean()
            .reset_index()
        )

    eval_single = df_single[
        (df_single["mode"] == "EVAL")
        & (df_single["twin_id"].str.contains("global", case=False))
    ].copy()
    if eval_single.empty:
        eval_single = (
            df_single[df_single["mode"] == "EVAL"]
            .groupby("round")["reward"]
            .mean()
            .reset_index()
        )

    # Sort by round
    eval_fl = eval_fl.sort_values("round")
    eval_single = eval_single.sort_values("round")

    # Plotting
    plt.figure(figsize=(12, 7))

    # Modern Styling
    plt.style.use("seaborn-v0_8-muted")  # Use a clean style

    plt.plot(
        eval_fl["round"],
        eval_fl["reward"],
        marker="o",
        linewidth=3,
        markersize=8,
        label="Fed-Twin (Global)",
        color="#2ecc71",
    )
    plt.plot(
        eval_single["round"],
        eval_single["reward"],
        marker="s",
        linewidth=3,
        markersize=8,
        label="Single-Twin (Baseline)",
        color="#e74c3c",
    )

    # Labels and details
    plt.title(
        "Performance Comparison: Fed-Twin vs. Single-Twin",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Average Reward (per episode)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12, frameon=True, shadow=True)

    # Annotate final values
    if not eval_fl.empty:
        plt.annotate(
            f"{eval_fl['reward'].iloc[-1]:.2f}",
            xy=(eval_fl["round"].iloc[-1], eval_fl["reward"].iloc[-1]),
            xytext=(5, 5),
            textcoords="offset points",
            color="#27ae60",
            fontweight="bold",
        )

    if not eval_single.empty:
        plt.annotate(
            f"{eval_single['reward'].iloc[-1]:.2f}",
            xy=(eval_single["round"].iloc[-1], eval_single["reward"].iloc[-1]),
            xytext=(5, -15),
            textcoords="offset points",
            color="#c0392b",
            fontweight="bold",
        )

    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    output_path = "plots/comparison_result.png"
    plt.savefig(output_path, dpi=300)
    print(f"✓ Comparison chart saved to: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        fl_path = sys.argv[1]
        single_path = sys.argv[2]
        plot_comparison(fl_path, single_path)
    else:
        plot_comparison()
