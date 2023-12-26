import time
from tqdm import tqdm, trange
from dask.distributed import wait
from simulate import simulate


def run_search(client, scheduler, env_seed, iterations, log_freq):
    """Runs the QD algorithm for the given number of iterations.

    Args:
        client (Client): A Dask client providing access to workers.
        scheduler (Scheduler): pyribs scheduler.
        env_seed (int): Seed for the environment.
        iterations (int): Iterations to run.
        log_freq (int): Number of iterations to wait before recording metrics.
    Returns:
        dict: A mapping from various metric names to a list of "x" and "y"
        values where x is the iteration and y is the value of the metric. Think
        of each entry as the x's and y's for a matplotlib plot.
    """
    print(
        "> Starting search.\n"
        "  - Open Dask's dashboard at http://localhost:8787 to monitor workers."
    )

    metrics = {
        "Max Score": {
            "x": [],
            "y": [],
        },
        "Archive Size": {
            "x": [0],
            "y": [0],
        },
    }

    start_time = time.time()
    for itr in trange(1, iterations + 1):
        # Request models from the scheduler.
        sols = scheduler.ask()

        # Evaluate the models and record the objectives and measures.
        objs, meas = [], []

        # Ask the Dask client to distribute the simulations among the Dask
        # workers, then gather the results of the simulations.
        futures = client.map(lambda model: simulate(model, env_seed), sols)
        results = client.gather(futures)
        wait(futures)

        # Process the results.
        for obj, impact_x_pos, impact_y_vel in results:
            objs.append(obj)
            meas.append([impact_x_pos, impact_y_vel])

        # Send the results back to the scheduler.
        scheduler.tell(objs, meas)

        # Logging.
        if itr % log_freq == 0 or itr == iterations:
            elapsed_time = time.time() - start_time
            metrics["Max Score"]["x"].append(itr)
            metrics["Max Score"]["y"].append(scheduler.archive.stats.obj_max)
            metrics["Archive Size"]["x"].append(itr)
            metrics["Archive Size"]["y"].append(len(scheduler.archive))
            tqdm.write(
                f"> {itr} itrs completed after {elapsed_time:.2f} s\n"
                f"  - Max Score: {metrics['Max Score']['y'][-1]}\n"
                f"  - Archive Size: {metrics['Archive Size']['y'][-1]}")

    return metrics