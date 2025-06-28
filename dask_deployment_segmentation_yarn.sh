#!/bin/bash -l
#SBATCH --job-name=segmentation_yarn
#SBATCH --time=12:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=1400G
#SBATCH -p nice

# Activate conda env
export PYTHONPATH=''
eval "$(/home/esrf/rofische/conda_x86/miniforge3/condabin/conda shell.bash hook)"
conda activate base

cd ~/lib/pytrainseg
git checkout yarn
dask scheduler --scheduler-file ~/scheduler_yarn.json  &
	sleep 10
	dask worker --nworkers 2 --memory-limit 780GB --scheduler-file ~/scheduler_yarn.json &
		sleep 10
		timeout 11h python /home/esrf/rofische/lib/pytrainseg/pick_up_crashed_segmentation.py
		if [[ $? == 124 ]]; then 
		  echo resubmit
		  sbatch dask_deployment_segmentation_yarn.sh
		fi
