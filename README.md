# PyTrainSeg - preliminary implementation of Weka-like 4D machine learning segmentation

## Description

This collection of codes extends the idea of trainable Weka segmentation (TWS, https://imagej.net/plugins/tws/) to 4D. The implementation is in Python using dask for lazy evaluation and juypter for interactive training. As any ML method, the 4D ML segmentation is computationally very heavy. Consider if it is really necessary.


## Usage
This is a project in development. There are three python classes for image feature creation, training and segmentation called from a main jupyter notebook. 
It might work if you execute the example notebook from top to bottom. Depending on your hardware and python environment, this might not work at all or crash eventually. 
I am working on a 64 cores 1 TB RAM 3TB SSD work station for my 4D-CT data (roughly 1700x150x800x70 voxel). I would start testing with a much smaller ROI.
Currently, the classes are not used as they have been layed out and some functionalities are not compatible.

## Main concepts

The 4D image data has to be loaded by dask itself from h5df. I like to have my image data as netcdf4 (a normed h5df) on a SSD. Dask creates chunkes (pieces) of the data which then allow the lazy loading and evaluation. Calculating features for the full dataset at once is way larger than available RAM.  As in TWS, a set of features is created for every pixel by applying a bunch of image filters. The new addition is the employment of 4D filters (e.g. 4D Gaussian Blur) and time specific filters (e.g. minimum grayvalue over all time steps). By manually labeling parts of the image, a random forest classifier is trained to assign every pixel to the label classes based on its feature set.

## Contact

The state of the project is basically a breadboard with wires sticking out and external devices patched with duct tape. The are too many limitations and bugs to list here at the moment. However, it works good enough for my inteded usage :). Feel free to use it, but I would really appreciate to learn of modifications you make since they might be helpful for me, too.
I see potential in a better technical implementation, as well as in the selection and employment of clever alternative image filters, esp. in the time domain.
If you want to dig into it, maybe you want to talk to me first for demonstration.


Cheers,

Robert Fischer

robert.fischer@psi.ch
