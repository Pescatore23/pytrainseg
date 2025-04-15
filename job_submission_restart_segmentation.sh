#!/bin/bash -l
#SBATCH --job-name=segmentation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -c 56
#SBATCH --mem=0
#SBATCH -p day
#SBATCH --exclusive


# Activate conda env
export PYTHONPATH=''
#conda activate membrane_fingering
eval "$(/das/home/fische_r/miniconda3/bin/conda shell.bash hook)"
conda activate base

# debugging flags (optional)
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun dask-scheduler --scheduler-file scheduler.json
surn dask-worker --nworkers=2 --memory-limit 160GB --scheduler-file scheduler.json

# Execute command in the container, ipython for debugging to avoid defaul python, change back to python eventually
srun python ~/lib/pytrainseg/pick_up_crashed_segmentation.py
