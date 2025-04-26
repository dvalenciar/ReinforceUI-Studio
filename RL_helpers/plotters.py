import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_comparison(
    algo_folders: dict[str, str], output_file: str, title: str = "Evaluation Comparison"
) -> None:
    """Plot evaluation logs from multiple algorithms for comparison.

    Args:
        algo_folders: Dict mapping algorithm display names (e.g. 'TD3 (1)') to their log folder paths.
        output_file: Path to save the final plot.
        title: Title for the comparison plot.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(12, 7), facecolor="#f5f5f5")

    for algo_name, folder_path in algo_folders.items():
        eval_path = os.path.join(folder_path, "data_log/evaluation_log.csv")
        if not os.path.exists(eval_path):
            print(f"[Warning] Evaluation log not found for {algo_name}")
            continue

        df = pd.read_csv(eval_path)
        df_grouped = df.groupby("Total Timesteps", as_index=False).last()

        sns.lineplot(
            x=df_grouped["Total Timesteps"],
            y=df_grouped["Average Reward"],
            label=algo_name,
            linewidth=2.5,
        )

    plt.title(title, fontsize=20, fontweight="bold")
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.gca().set_facecolor("#eaeaf2")
    plt.legend(fontsize=12)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()


def plot_logs(
    logs: list,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    output_file: str,
) -> None:
    """Plot logs and save the plot as an image file.

    Args:
        logs: List of logs to be plotted.
        x_column: The column name for the x-axis.
        y_column: The column name for the y-axis.
        title: The title of the plot.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
        output_file: The name of the output image file.
    """
    df = pd.DataFrame(logs)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Group data by the total timesteps and get the last average reward for each group
    # this is to avoid plotting multiple points for the same total timesteps in evaluation logs
    df_grouped = df.groupby("Total Timesteps", as_index=False).last()

    plt.figure(figsize=(10, 6), facecolor="#f5f5f5")
    plt.title(title, fontsize=20, fontweight="bold")
    sns.lineplot(
        x=df_grouped[x_column],
        y=df_grouped[y_column],
        linewidth=2.5,
        color="r",
    )
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.gca().set_facecolor("#eaeaf2")
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
