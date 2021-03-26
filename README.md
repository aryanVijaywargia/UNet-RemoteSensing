# UNet Agriculture Segmentation

### Model Training 
The training images and the corresponding output array were reshaped into a specific matrix format, e.g., (batch_size, band_count, width, height, classes) and then mapped pixel by pixel.


## Model Architecture
![UNet Architecture](model.png)
### Information about the various code files
* config.py file includes some properties like dataset directory, test directory and the colormap.
* make_dataset.py file is used for feature extraction and creating dataset.
* train_model.py contains the UNet Architecture
* utils.py file contains the helper functions

One of the most challenging jobs in training the U-Net is streaming the images into U-Net. There are certain ways to do so. Python provides very easy-to-use method to read multiple-dimension arrays from image files. The arrays (numpy arrays) will be reshaped into a specific matrix format, e.g., (batch_size, band_count, width, height, classes). The corresponding output array will be reshaped into some similar shapes with categorized probability array. The mapping between input arrays and output arrays must be exactly matched pixel by pixel. Otherwise, the training will be void.