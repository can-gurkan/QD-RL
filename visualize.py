import json
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
