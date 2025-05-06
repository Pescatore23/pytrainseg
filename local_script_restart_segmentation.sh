conda activate
dask scheduler --scheduler-file ~/scheduler.json &
sleep 10
dask worker --nworkers=2 --memory-limit 160GB --scheduler-file ~/scheduler.json &
sleep 120
python ~/lib/pytrainseg/pick_up_crashed_segmentation.py
