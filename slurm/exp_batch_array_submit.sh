#!/bin/bash

CONFIG="$1"
EXPNAME="$2"
EXPPARAMS="$3"
REPS="$4"

for (( ri=1; ri<=$REPS; ri++ )); do
    sbatch slurm/exp_array_submit.sh $CONFIG $EXPNAME $EXPPARAMS $ri
    sleep 4;
done
