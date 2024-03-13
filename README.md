# QD-RL

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

QD-RL is a codebase for investigating the impact of open-endedness of search algorithms on representation.

## Installation

**Clone the repo:**
   ```bash
   git clone https://github.com/can-gurkan/QD-RL.git
   ```
Install the dependencies (see requirements.txt)
We recommend using a conda virtual environment with python 3.11.
Install the following to get started:
- pytorch
- pyribs-visualize
- dask
- distributed
- fire
- gin-config
- gymnasium
- gymnasium-box2d
- gymnasium-mujoco
- minigrid
- swig
- tqdm

```sh
conda install -c pytorch pytorch
conda install -c conda-forge pyribs-visualize
conda install -c conda-forge dask
conda install -c conda-forge distributed
conda install -c conda-forge fire
pip install gin-config
conda install -c conda-forge gymnasium
conda install -c conda-forge gymnasium-box2d
conda install -c conda-forge gymnasium-mujoco
pip install minigrid
conda install -c anaconda swig
conda install -c conda-forge tqdm
```

## Usage

You can run experiments from the commandline using:
```sh
python main.py --config_file=config/<my_config.gin> --exp_name=<"my_exp"> --archive_size=<"(n,n)"> --reps=<"n">
```
If archive no arguments are provided, a test experiment will be run according to the parameters in config/hyperparams_test.gin

To run on a slurm cluster use:
```sh
bash ./slurm/exp_batch_array_submit.sh config/<my_config.gin> <"my_exp"> config/<my_exp_vars.txt> <num-reps>
```
This will create the necessary sub-directories and run num-reps repetitions of all the archive-size parameters provided in my_exp_vars.txt and use the parameters in my_config.gin for everything else.

### Configuring
To configure the experiments use .gin files in the config directory.
The main parameters are:

 experiment.iterations: Number of training iterations  
 experiment.workers: Number of Dask workers running in parallel  
 experiment.env_seed: Random seed for the environment  
 experiment.log_freq: Logging frequency  
 experiment.n_emitters: Number of emitters that determines the 
 number of instances of ME that mutates solutions  
 experiment.batch_size: Batch size for the number of mutations for each emitter  
 experiment.sigma0: 1.0  
 experiment.seed: Random seed of the entire experiment  
 experiment.outdir: The path for the output directory of experiment outcomes  

 create_scheduler.archive_type: The type of archive for ME (e.g. Grid or CVT) 
 GridArchive.dims: A tuple that determines the resolution of the archive  
 GridArchive.ranges: A list of tuples that determines the range of each dimension  
 GridArchive.learning_rate: 1.0  
 GridArchive.qd_score_offset: Used to get positive values when calculating the QD score  

 create_scheduler.emitter_type: The type of emitter that mutates solutions (e.g. ES emitter)
 create_scheduler.env_name: Name of the environment or domain
 EvolutionStrategyEmitter.ranker: The ranking method used to prioritize mutated solutions
 
 MLP.layer_shapes: A tuple determining the number of hidden layers and the number of nodes per layer

## Examples
Run a test experiment with default values in config/hyperparams_test.gin
```sh
python main.py
```
Run an experiment on the QD-HalfCheetah environment for 5000 training iterations with a neural net with hidden layers size (8,8) using the parameters in config/hparams_exp_hc_n8.gin under an experiment directory "exp_hc_n8_itr5000" for all the archive dimensions listed in config/expvars_hc_n8.txt each with 10 repetitions.
```sh
bash ./slurm/exp_batch_array_submit.sh config/hparams_exp_hc_n8.gin "exp_hc_n8_itr5000" config/expvars_hc_n8.txt 10
```

## Heavily Used Libraries

To understand the code, it will be useful to be familiar with the following libraries:

- [pytorch](https://pytorch.org/)
- [pyribs](https://pyribs.org)
- [dask](https://dask.org)
- [gin](https://github.com/google/gin-config)

## Docker

## License

MIT

