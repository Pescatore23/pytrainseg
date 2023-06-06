# PyTrainSeg - preliminary implementation of Weka-like 4D machine learning segmentation

## Description

This collection of codes extends the idea of trainable Weka segmentation (TWS, https://imagej.net/plugins/tws/) to 4D. The implementation is in Python using dask for lazy evaluation and juypter for interactive training. As any ML method, the 4D ML segmentation is computationally very heavy. Consider if it is really necessary.


## Usage
This is a project in development. There are three python classes for image feauture creation, training and segmentation called from a main jupyter notebook. Currently, the classes are not used as they have been layed out and some functionalities are not compatible, for example the planned usage of lazy loading is not working.
It might work, if you execute the example notebook from top to bottom. Depending on your hardware, this might not work at all or crash eventually. I share this code in hope that parts might be useful and to receive feedback to make it more stable. Also, clever alternative image filters are highly appreciated.

## Main concepts

The 4D image data is available in RAM in its entirety as dask array with appropiate chunks. The chunks then allow the lazy evaluation, because calculating features for the full dataset at once is way larger than available RAM. I like to have my image data as netcdf4 (a normed h5df) on disk, but as long 
as you can create a 4D dask array, you will be fine. As in TWS, a set of features is created for every pixel by applying a bunch of image filters. The new addition is the employment of 4D filters (e.g. Gaussian) and time specific filters (e.g. minimum grayvalue over all time steps). By manually labeling parts of the image, a random forest classifier is trained to 
assign every pixel to the label classes based on its feature set.

## Contact

The state of the project is basically a breadboard with wires sticking out and external devices patched with duct tape. The are to many liomitations and bugs to list here at the moment. However, it works :). Feel free to use it, but I would really appreciate to learn about modifications you make since they might be helpful for me, too.
If you want to dig into it, maybe you want to talk to me.


Cheers,

Robert Fischer

robert.fischer@psi.ch