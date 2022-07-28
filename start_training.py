from ij import IJ
import os

time_step = 5
slice_number = 54

slice_name= ''.join(['ts_',str(time_step),'_slice_',str(slice_number)])

#training_path = r"U:\01_Python\00_playground\test_pytorch\Dataset\test_tomcat\training"
#XTM_data_path = r"D:\TOMCAT_2\01_intcorrect_med_leg_0"

XTM_data_path = r"C:\Zwischenlager\wood_time_slices\00_raw"
training_path = r"C:\Zwischenlager\wood_time_slices\training_data"


time_folder = os.listdir(XTM_data_path)
timestep_folder = time_folder[time_step]
images = os.listdir(os.path.join(XTM_data_path, timestep_folder))
image_name = images[slice_number]

watername = ''.join([slice_name, '_water.tif'])
waterpath = os.path.join(training_path, watername)
airname = ''.join([slice_name, '_air.tif'])
airpath = os.path.join(training_path, airname)
fibername = ''.join([slice_name, '_fiber.tif'])
fiberpath = os.path.join(training_path, fibername)
resultname = ''.join([slice_name, '_classified.tif'])
resultpath = os.path.join(training_path, resultname)

# open raw image
im = IJ.openImage(os.path.join(XTM_data_path, timestep_folder, image_name))
im.show()
#check if there are already ground truth images 
#then open or create

if os.path.exists(airpath):
	air = IJ.openImage(airpath)
else:
	air = IJ.createImage(airname, "8-bit black",  im.width, im.height, 1);
	IJ.saveAs(air, "Tiff", airpath)
air.show()

if os.path.exists(waterpath):
	water = IJ.openImage(waterpath)
else:
	water = IJ.createImage(watername, "8-bit black",  im.width, im.height, 1);
	IJ.saveAs(water, "Tiff", waterpath)
water.show()

if os.path.exists(fiberpath):
	fiber = IJ.openImage(fiberpath)
else:
	fiber = IJ.createImage(fibername, "8-bit black",  im.width, im.height, 1);
	IJ.saveAs(fiber, "Tiff", fiberpath)
fiber.show()

if os.path.exists(resultpath):
	result = IJ.openImage(resultpath)
	result.show()
