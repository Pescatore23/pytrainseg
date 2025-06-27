#!/bin/bash -l
#SBATCH --job-name=segmentation_yarn
#SBATCH --time=12:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=1400G
#SBATCH -p nice

# Activate conda env
export PYTHONPATH=''
#eval "$(/store/empa/em13/fischer/lib/miniconda3/bin/conda shell.bash hook)"
#conda activate membrane_fingering
eval "$(/home/esrf/rofische/conda_x86/miniforge3/condabin/conda shell.bash hook)"
conda activate base

cd ~/lib/pytrainseg
git checkout yarn
cd
dask scheduler --scheduler-file ~/scheduler_yarn.json  &
	sleep 5
	dask worker --nworkers 2 --memory-limit 780GB --scheduler-file ~/scheduler_yarn.json &
		sleep 5
		python /home/esrf/rofische/lib/pytrainseg/pick_up_crashed_segmentation.py
