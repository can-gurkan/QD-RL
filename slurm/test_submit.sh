#!/bin/bash
#SBATCH --account=p20016  ## YOUR ACCOUNT
#SBATCH --partition=gengpu  ### PARTITION (short: 0-4 hours, normal: 4-48 hours, long: >48 hours w10001, etc)
#SBATCH --gres=gpu:a100:1
###SBATCH --array=0-9 ## number of jobs to run "in parallel" 
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=4 ## how many cpus or processors do you need on each computer
#SBATCH --time=00:10:00 ## <hh:mm:ss> how long does this need to run 
#SBATCH --mem-per-cpu=1G ## how much RAM do you need per CPU (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name="qdrl_test1" ## use the task id in the name of the job
#SBATCH --output=/projects/p20016/Can/qdrl_outputs/quest_info.txt ##sample_job.%A_%a.out ## use the jobid (A) and the specific job index (a) to name your log file
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=gurkan@u.northwestern.edu  ## your email

module purge all
module load python/anaconda3.6
source activate qdrl_env2

python main.py
