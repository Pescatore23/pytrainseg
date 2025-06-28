#!/bin/bash -l
#SBATCH --job-name=segmentation_yarn
#SBATCH --time=12:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH -p nice
#SBATCH --exclusive

# Activate conda env
export PYTHONPATH=''
eval "$(/home/esrf/rofische/conda_x86/miniforge3/condabin/conda shell.bash hook)"
conda activate base

dask scheduler --scheduler-file ~/scheduler_yarn.json  &
	sleep 10
	dask worker --nworkers 2 --memory-limit 780GB --scheduler-file ~/scheduler_yarn.json &
		sleep 10
		timeout 11h python /home/esrf/rofische/lib/pytrainseg/pick_up_yarn_segmentation.py
#		timeout 11h python /home/esrf/rofische/lib/python_playground/test_dask_script.py
		if [[ $? == 124 ]]; then 
		  echo resubmit
		  sbatch dask_deployment_segmentation_yarn.sh
		fi
