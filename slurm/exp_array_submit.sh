#!/bin/bash
#SBATCH --account=p20016  ## YOUR ACCOUNT
#SBATCH --partition=normal  ### PARTITION (gengpu, short: 0-4 hours, normal: 4-48 hours, long: >48 hours w10001, etc)
###SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2 ## number of jobs to run "in parallel" 
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=50 ## how many cpus or processors do you need on each computer

###SBATCH --constraint=quest12
#SBATCH --time=20:50:00 ## <hh:mm:ss> how long does this need to run 
###SBATCH --mem=0
#SBATCH --mem-per-cpu=1500M ##1500M  ##750M  ##1G ## how much RAM do you need per CPU (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name="qdrl_\${SLURM_ARRAY_TASK_ID}" ## use the task id in the name of the job
#SBATCH --output=/projects/p20016/Can/qdrl_outputs/output_logs/qinfo.%A_%a.txt ##sample_job.%A_%a.out ## use the jobid (A) and the specific job index (a) to name your log file
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=gurkan@u.northwestern.edu  ## your email

CONFIG="$1"
EXPNAME="$2"
EXPPARAMS="$3"
REP_ID="${4:NULL}"

module purge all
module load python/anaconda3.6
source activate qdrl_env2

IFS=$'\n' read -d '' -r -a expvars < $EXPPARAMS

if [[ $REP_ID -eq "NULL" ]]; then
    python main.py --config_file=$CONFIG --exp_name=$EXPNAME --archive_size=${expvars[$SLURM_ARRAY_TASK_ID]}
else
    python main.py --config_file=$CONFIG --exp_name=$EXPNAME --archive_size=${expvars[$SLURM_ARRAY_TASK_ID]} --reps=$REP_ID
fi
