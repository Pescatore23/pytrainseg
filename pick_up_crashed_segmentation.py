# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:22:01 2025

TODO: make nice, currently just copy of the relevant jupyter cells without comments

@author: ROFISCHE
"""

# modules
import os
import xarray as xr
import numpy as np
import dask
import dask.array
import pickle

from dask.distributed import Client #, LocalCluster
import socket
import subprocess
import h5py
import logging
import warnings
warnings.filterwarnings('ignore')
import json


ESRFhome = '/home/esrf/rofische'

from dask import config as cfg
# cfg.set({'distributed.scheduler.worker-ttl': None, # Workaround so that dask does not kill workers while they are busy fetching data: https://dask.discourse.group/t/dask-workers-killed-because-of-heartbeat-fail/856, maybe this helps: https://www.youtube.com/watch?v=vF2VItVU5zg?
#         'distributed.scheduler.transition-log-length': 100, #potential workaround for ballooning scheduler memory https://baumgartner.io/posts/how-to-reduce-memory-usage-of-dask-scheduler/
#          'distributed.scheduler.events-log-length': 100
#         }) seems to be outdate

cfg.set({'distributed.scheduler.worker-ttl': None, # Workaround so that dask does not kill workers while they are busy fetching data: https://dask.discourse.group/t/dask-workers-killed-because-of-heartbeat-fail/856, maybe this helps: https://www.youtube.com/watch?v=vF2VItVU5zg?
        'distributed.admin.low-level-log-length': 100 #potential workaround for ballooning scheduler memory https://baumgartner.io/posts/how-to-reduce-memory-usage-of-dask-scheduler/
        }) # still relevant ?

#paths
host = socket.gethostname()
print(host)
if host == 'mpc2959.psi.ch':
    temppath = '/mnt/SSD/fische_r/tmp'
    training_path =  '/mnt/SSD/fische_r/Tomcat_2/'
    pytrainpath = '/mpc/homes/fische_r/lib/pytrainseg'
    # memlim = '840GB'
    memlim = '440GB'
    # memlim = '920GB'
    home = '/mpc/homes/fische_r'
elif host[:3] == 'ra-':
    temppath = '/das/home/fische_r/interlaces/Tomcat_2/tmp'
    training_path = '/das/home/fische_r/interlaces/Tomcat_2'
    pytrainpath = '/das/home/fische_r/lib/pytrainseg'
    memlim = '160GB'  # also fine on the small nodes, you can differentiate more if you want
    home = rahome
elif host[:3] == 'hpc': # 
    temppath = '/tmp/robert'
    training_path = '/home/esrf/rofische/data_ihma664/PROCESSED_DATA/TOMCAT/Tomcat_2'
    pytrainpath = '/home/esrf/rofische/lib/pytrainseg'
    memlim = '780GB'
    home = ESRFhome 
else:
    print('host '+host+' currently not supported')
    
scheduler_dict = json.load(open(os.path.join(home, 'scheduler.json'),'r'))
scheduler_address = scheduler_dict['address']
# get the ML functions, TODO: make a library once it works/is in a stable state


cwd = os.getcwd()
os.chdir(pytrainpath)
from feature_stack import image_filter
from training import training

pytrain_git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
os.chdir(cwd)

######## parse some arguments
######## need to be consitent with original jupyter notebook
sample = 'R_m4_33_050_2'
prefix = '2025-06-26_git_sha_59a9df3' #for classifier filepath
dim1 = 52 #better use multiple of chunk size !?  <-- tune this parameter to minimize imax, jmax and the size of the result
#################

# feature_names_to_use = ['Gaussian_4D_Blur_0.0',
 # 'Gaussian_4D_Blur_1.0',
 # 'Gaussian_4D_Blur_6.0',
 # 'diff_of_gauss_4D_6.0_0.0',
 # 'diff_of_gauss_4D_6.0_1.0',
 # 'Gradient_sigma_1.0_0',
 # 'Gradient_sigma_1.0_1',
 # 'Gradient_sigma_1.0_3',
 # 'hessian_sigma_1.0_00',
 # 'hessian_sigma_1.0_01',
 # 'hessian_sigma_1.0_11',
 # 'Gradient_sigma_3.0_3',
 # 'hessian_sigma_3.0_00',
 # 'hessian_sigma_3.0_01',
 # 'hessian_sigma_3.0_02',
 # 'hessian_sigma_3.0_03',
 # 'hessian_sigma_3.0_11',
 # 'hessian_sigma_3.0_33',
 # 'Gradient_sigma_6.0_0',
 # 'Gradient_sigma_6.0_1',
 # 'Gradient_sigma_6.0_2',
 # 'Gradient_sigma_6.0_3',
 # 'hessian_sigma_6.0_01',
 # 'hessian_sigma_6.0_03',
 # 'hessian_sigma_6.0_11',
 # 'hessian_sigma_6.0_12',
 # 'hessian_sigma_6.0_13',
 # 'hessian_sigma_6.0_22',
 # 'hessian_sigma_6.0_23',
 # 'hessian_sigma_6.0_33',
 # 'Gradient_sigma_2.0_0',
 # 'Gradient_sigma_2.0_3',
 # 'hessian_sigma_2.0_00',
 # 'hessian_sigma_2.0_01',
 # 'hessian_sigma_2.0_02',
 # 'hessian_sigma_2.0_11',
 # 'hessian_sigma_2.0_13',
 # 'Gaussian_time_0.0',
 # 'Gaussian_time_1.0',
 # 'Gaussian_time_6.0',
 # 'Gaussian_time_2.0',
 # 'Gaussian_space_0.0',
 # 'Gaussian_space_6.0',
 # 'diff_of_gauss_space_3.0_0.0',
 # 'diff_of_gauss_space_6.0_0.0',
 # 'diff_of_gauss_space_2.0_0.0',
 # 'diff_of_gauss_space_3.0_1.0',
 # 'diff_of_gauss_space_6.0_1.0',
 # 'diff_of_gauss_space_2.0_1.0',
 # 'diff_of_gauss_space_2.0_3.0',
 # 'diff_temp_min_Gauss_2.0',
 # 'diff_to_first_',
 # 'full_temp_min_Gauss_2.0',
 # 'first_',
 # 'last_']

#############



dask.config.config['temporary-directory'] = temppath
def boot_client(dashboard_address=':35000', memory_limit = memlim, n_workers=2, scheduler_address = scheduler_address): # 2 workers appears to be the optimum, will still distribute over the full machine
    # tempfolder = temppath  #a big SSD is a major adavantage to allow spill to disk and still be efficient. large dataset might crash with too small SSD or be slow with normal HDD
    #cluster = LocalCluster(dashboard_address=dashboard_address, memory_limit = memory_limit, n_workers=n_workers, silence_logs=logging.ERROR) #settings optimised for mpc2959, play around if needed, if you know nothing else is using RAM then you can almost go to the limit
    #client = Client(cluster) #don't show warnings, too many seem to block execution
    client = Client(scheduler_address)
    print('Dashboard at '+client.dashboard_link)
    return client #, cluster

def reboot_client(client, dashboard_address=':35000', memory_limit = memlim, n_workers=2, scheduler_address = scheduler_address):
    client.shutdown()
    #cluster = LocalCluster(dashboard_address=dashboard_address, memory_limit = memory_limit, n_workers=n_workers, silence_logs=logging.ERROR)
    #client = Client(cluster)
    client = Client(scheduler_address)
    return client

#client, cluster = boot_client()
client = boot_client()

imagepath = os.path.join(training_path, '01_'+sample+'_cropped_filtered.nc')
file = h5py.File(imagepath)

chunk_space = dim1 # potential for optmisation by matching chunksize with planned image filter kernels and file structure on disk for fast data streaming
chunks = (chunk_space,chunk_space,chunk_space,len(file['timestep']))
da = dask.array.from_array(file['image_data'], chunks= chunks)

IF = image_filter(sigmas = [0,1,3,6]) 
IF.data = da
shp = da.shape
shp_raw = shp

IF.prepare()
IF.stack_features()
#IF.reduce_feature_stack(feature_names_to_use)
IF.make_xarray()

training_path_sample = os.path.join(training_path, sample)
if not os.path.exists(training_path_sample):
    os.mkdir(training_path_sample)

TS = training(training_path=training_path_sample)
TS.client = client
IF.client = client
#TS.cluster = cluster
#IF.cluster = cluster
TS.memlim = memlim
TS.n_workers = 2

TS.feat_data = IF.feature_xarray
IF.combined_feature_names = list(IF.feature_names) + list(IF.feature_names_time_independent)
TS.combined_feature_names = IF.combined_feature_names

clf = pickle.load(open(os.path.join(training_path, prefix+'_clf.p'),'rb'))
clf.n_jobs = 64
feat = TS.feat_data['feature_stack']
feat_idp = TS.feat_data['feature_stack_time_independent']

def round_up(val, dec=1):
    rval = np.round(val, dec)
    if rval < val:
        rval = rval+10**(-dec)
    rval = np.round(rval, dec) # to get rid of floating point uncertainties
    return rval

def calculate_part(part):
    if type(part) is not np.ndarray:
        fut = client.scatter(part)
        fut = fut.result()
        fut = fut.compute()
        part = fut
    return part

def get_the_client_back(client):
    try:
        client.restart()
    except:
        client = reboot_client(client)
    if not len(client.cluster.workers)>1:   
        client = reboot_client(client)
    return client

# shp = feat.shape[:-1]
shp = shp_raw

# aspect ratio 
dimsize = np.sort(shp[:-1] )
aspect = round_up(dimsize[-1]/dimsize[-2])

# check length of loops to process entire dataset, estimate size of obtained sub-feature to avoid out-of-memory issues

dim2 = int(round_up(aspect*dim1, 0))
jmax = int(round_up(dimsize[-2]/dim1, 0))
imax = int(round_up(dimsize[-1]/dim2, 0))

piecepath = os.path.join(os.path.join(training_path_sample, 'segmentation_pieces'))

restart_i = 0
restart_j = 0  #replace with the iterations you coudl reach before dask crashed
# get from the written files the restart coordinates
print(piecepath)
if os.path.exists(piecepath):
    ij_mat = np.zeros((imax,jmax), dtype=bool)
    for filename in os.listdir(piecepath):
        i = int(filename.split('_')[2])
        j = int(filename.split('_')[4])
        ij_mat[i,j] = True
        restart_i = int(np.min(np.argmin(ij_mat, axis=0)))
        restart_j = int(np.max(np.argmin(ij_mat, axis=1)))


print('restarting at:',restart_i, restart_j)

# restart_i = 0
# restart_j = 14

# elapsed walltime: subprocess.check_output(['squeue','-u', 'fische_r']).decode().strip().split(' ')[-8]

# piecepath = os.path.join(os.path.join(temppath, 'segmentation_pieces'))
if not os.path.exists(piecepath):
    os.mkdir(piecepath)

for i in range(restart_i,imax):
    print(str(i+1)+'/'+str(imax))
    start_j = 0
    if i == restart_i:
        start_j = restart_j
    for j in range(start_j,jmax):
        print(j)
        tmpfile = os.path.join(piecepath, 'seg_i_'+str(i)+'_j_'+str(j)+'_.p')
        if os.path.exists(tmpfile):
            print(i,j,'already exists, skipping')
            continue
        part = feat[i*dim1:(i+1)*dim1,:,j*dim2:(j+1)*dim2,:,:] 
        part_idp = feat_idp[i*dim1:(i+1)*dim1,:,j*dim2:(j+1)*dim2,:]  
        if 0 in part.shape:
            print('hit the edge (one dimension 0), ignore')
            continue
        part = calculate_part(part)
        #client = get_the_client_back(client)
        part_idp = calculate_part(part_idp)
        #client = get_the_client_back(client)
        
        part_idp = np.stack([part_idp]*shp[-1], axis=-2)[:,:,:,0,:,:] #expand in time, a bit ugly, could maybe more elegant
        part = np.concatenate([part, part_idp], axis = -1)
        del part_idp # drop the time independent part, is this garbage collected?

        shp_orig = part.shape
        num_feat = part.shape[-1]  
        part = part.reshape(-1,num_feat)

        seg = clf.predict(part).astype(np.uint8)

        # put segs together when all calculated
        seg = seg.reshape(shp_orig[:4])

        pickle.dump(seg, open(tmpfile, 'wb'))
        
segs = np.zeros(shp_raw, dtype=np.uint8)

for filename in os.listdir(piecepath):
    i = int(filename.split('_')[2])
    j = int(filename.split('_')[4])
    ### not sure if this switch cases are necessary
    print(filename)
    seg = pickle.load(open(os.path.join(piecepath, filename),'rb'))
    if i < imax-1 and j < jmax-1:
        segs[i*dim1:(i+1)*dim1,:,j*dim2:(j+1)*dim2,:] = seg
    elif not i < imax-1 and j < jmax-1:
        segs[i*dim1:,:,j*dim2:(j+1)*dim2,:] =  seg
    elif not j < jmax-1 and i < imax-1:
        segs[i*dim1:(i+1)*dim1,:,j*dim2:,:] =  seg
    else:
        segs[i*dim1:,:,j*dim2:,:] = seg
        
        
# TODO: add metadat and time data from original dataset   
shp = segs.shape
segdata = xr.Dataset({'segmented': (['x','y','z','timestep'], segs),},
                               coords = {'x': np.arange(shp[0]),
                               'y': np.arange(shp[1]),
                               'z': np.arange(shp[2]),
                               'timestep': np.arange(shp[3]),
                               'feature': TS.combined_feature_names}
                     )
segdata.attrs['pytrain_git'] = pytrain_git_sha

segpath = os.path.join(training_path_sample, sample+'_segmentation.nc')
segdata.to_netcdf(segpath)
