from pathlib import Path
from dask.distributed import Client, LocalCluster
import fire
import gin
import time

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from scheduler import create_scheduler
from search import run_search
from visualize import *

@gin.configurable
def experiment(workers=8,
        env_seed=52,
        iterations=300,
        log_freq=25,
        n_emitters=5,
        batch_size=50,
        sigma0=1.0,
        seed=None,
        outdir="output_files"):
    
    """
    Args:
        workers (int): Number of workers to use for simulations.
        env_seed (int): Environment seed.
        iterations (int): Number of iterations to run the algorithm.
        log_freq (int): Number of iterations to wait before recording metrics and saving heatmap.
        n_emitters (int): Number of emitters.
        batch_size (int): Batch size of each emitter.
        sigma0 (float): Initial step size of each emitter.
        seed (seed): Random seed for the pyribs components.
        outdir (str): Directory for the output files.
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

    # Specify QD algorithm and run search.
    scheduler = create_scheduler(seed, n_emitters, sigma0, batch_size)
    metrics = run_search(client, scheduler, env_seed, iterations, log_freq)

    # Outputs.
    scheduler.archive.data(return_type='pandas').to_csv(outdir / "archive.csv")
    save_ccdf(scheduler.archive, str(outdir / "archive_ccdf.png"))
    # Fix this later to determine which heatmap to use based on env
    #save_cvt_heatmap(scheduler.archive, str(outdir / "heatmap.png"))
    save_heatmap(scheduler.archive, str(outdir / "heatmap.png"))
    save_metrics(outdir, metrics)
    make_video(outdir,env_seed)


def manager(exp_name='exp_test',**kwargs):
    curr_outdir = gin.query_parameter('experiment.outdir')
    archive_type = gin.query_parameter('create_scheduler.archive_type')
    if 'archive_size' in kwargs:
        archive_size = kwargs['archive_size']
        if 'GridArchive' in str(archive_type):
            outdir = curr_outdir + '/exps/' + exp_name + '_as_' + str(archive_size[0]**2)
            gin.bind_parameter('GridArchive.dims', archive_size)
            #print(gin.query_parameter('GridArchive.dims'))
        elif 'CVTArchive' in str(archive_type):
            outdir = curr_outdir + '/exps/' + exp_name + '_as_' + str(archive_size)
            gin.bind_parameter('CVTArchive.cells', archive_size)
            #print(gin.query_parameter('CVTArchive.cells'))
        experiment(outdir=outdir)
    else:
        print("No experiment parameters provided.")
    #print(outdir,archive_type)
    

def main(config_file='config/hyperparams_test.gin', exp_name=None, **kwargs):
    from ribs.archives import CVTArchive, GridArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from models import MLP

    gin.external_configurable(CVTArchive)
    gin.external_configurable(GridArchive)
    gin.external_configurable(EvolutionStrategyEmitter)
    gin.parse_config_file(config_file)

    if exp_name is not None:
        manager(exp_name=exp_name,**kwargs)
    else:
        experiment(iterations=10)
    #experiment()
    #experiment(iterations=1000)
    #experiment(workers=8,iterations=100000)
    #manager(exp_name='exp_test2')

if __name__ == "__main__":
    fire.Fire(main)
