import os
from pathlib import Path
import fire
from dask.distributed import Client, LocalCluster

from scheduler import create_scheduler
from search import run_search
from visualize import *

def main(workers=4,
        env_seed=52,
        iterations=300,
        log_freq=25,
        n_emitters=5,
        batch_size=30,
        sigma0=1.0,
        seed=None,
        outdir="output_files"):
    
    """Uses CMA-ME to train linear agents in Lunar Lander.

    Args:
        workers (int): Number of workers to use for simulations.
        env_seed (int): Environment seed. The default gives the flat terrain
            from the tutorial.
        iterations (int): Number of iterations to run the algorithm.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        n_emitters (int): Number of emitters.
        batch_size (int): Batch size of each emitter.
        sigma0 (float): Initial step size of each emitter.
        seed (seed): Random seed for the pyribs components.
        outdir (str): Directory for Lunar Lander output.
        run_eval (bool): Pass this flag to run an evaluation of 10 random
            solutions selected from the archive in the `outdir`.
    """
    outdir = Path(outdir)

    # Make the directory here so that it is not made when running eval.
    outdir.mkdir(parents=True,exist_ok=True)

    # Setup Dask. The client connects to a "cluster" running on this machine.
    # The cluster simply manages several concurrent worker processes. If using
    # Dask across many workers, we would set up a more complicated cluster and
    # connect the client to it.
    cluster = LocalCluster(
        processes=True,  # Each worker is a process.
        n_workers=workers,  # Create this many worker processes.
        threads_per_worker=1,  # Each worker process is single-threaded.
    )
    client = Client(cluster)

    # CMA-ME.
    scheduler = create_scheduler(seed, n_emitters, sigma0, batch_size)
    metrics = run_search(client, scheduler, env_seed, iterations, log_freq)

    # Outputs.
    scheduler.archive.as_pandas().to_csv(outdir / "archive.csv")
    save_ccdf(scheduler.archive, str(outdir / "archive_ccdf.png"))
    save_heatmap(scheduler.archive, str(outdir / "heatmap.png"))
    save_metrics(outdir, metrics)


if __name__ == "__main__":
    fire.Fire(main)