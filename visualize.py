import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap
from ribs.visualize import cvt_archive_heatmap


def save_heatmap(archive, filename):
    """Saves a heatmap of the scheduler's archive to the filename.

    Args:
        archive (GridArchive): Archive with results from an experiment.
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=-300, vmax=300, ax=ax)
    ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
    ax.set_ylabel("Impact y-velocity")
    ax.set_xlabel("Impact x-position")
    fig.savefig(filename)

def save_cvt_heatmap(archive, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    cvt_archive_heatmap(archive, lw=0.1, ax=ax)
    ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
    ax.set_ylabel("Impact y-velocity")
    ax.set_xlabel("Impact x-position")
    fig.savefig(filename)
    plt.figure(figsize=(12, 9))


def save_metrics(outdir, metrics):
    """Saves metrics to png plots and a JSON file.

    Args:
        outdir (Path): output directory for saving files.
        metrics (dict): Metrics as output by run_search.
    """
    # Plots.
    for metric in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics[metric]["x"], metrics[metric]["y"])
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        fig.savefig(str(outdir / f"{metric.lower().replace(' ', '_')}.png"))

    # JSON file.
    with (outdir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


def save_ccdf(archive, filename):
    """Saves a CCDF showing the distribution of the archive's objectives.

    CCDF = Complementary Cumulative Distribution Function (see
    https://en.wikipedia.org/wiki/Cumulative_distribution_function#Complementary_cumulative_distribution_function_(tail_distribution)).
    The CCDF plotted here is not normalized to the range (0,1). This may help
    when comparing CCDF's among archives with different amounts of coverage
    (i.e. when one archive has more cells filled).

    Args:
        archive (GridArchive): Archive with results from an experiment.
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots()
    ax.hist(
        archive.as_pandas(include_solutions=False)["objective"],
        50,  # Number of cells.
        histtype="step",
        density=False,
        cumulative=-1)  # CCDF rather than CDF.
    ax.set_xlabel("Objectives")
    ax.set_ylabel("Num. Entries")
    ax.set_title("Distribution of Archive Objectives")
    fig.savefig(filename)


def make_video(outdir, env_seed, best_n=5):
    """Simulates the best archive solutions and saves videos of them.

    Videos are saved to outdir / videos.
    Args:
        outdir (Path): Path object for the output directory from which to
            retrieve the archive and save videos.
        env_seed (int): Seed for the environment."""
    
    import gymnasium as gym
    from simulate import simulate

    df = pd.read_csv(outdir / "archive.csv")
    high_perf_sols = df.sort_values("objective", ascending=False)
    indices = high_perf_sols.iloc[0:best_n].index

    # Use a single env so that all the videos go to the same directory.
    video_env = gym.wrappers.RecordVideo(
        gym.make("LunarLander-v2", render_mode="rgb_array"),
        video_folder=str(outdir / "videos"),
        # This will ensure all episodes are recorded as videos.
        episode_trigger=lambda idx: True,
        # Disables moviepy's logger to reduce clutter in the output.
        disable_logger=True,
    )

    for idx in indices:
        model = np.array(df.loc[idx, "solution_0":])
        reward, impact_x_pos, impact_y_vel = simulate(model, env_seed, video_env)

        print(f"=== Index {idx} ===\n"
              "Model:\n"
              f"{model}\n"
              f"Reward: {reward}\n"
              f"Impact x-pos: {impact_x_pos}\n"
              f"Impact y-vel: {impact_y_vel}\n")
    
    video_env.close()  # Save video.