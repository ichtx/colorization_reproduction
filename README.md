# colorization_reproduction

## Reproduction of the Paper Colorful Image Colorization

### Original Github Repo Codes
Using demo_release.py from original codes, we can colorize images using the ECCV model.

### Our Implementation
As we perform classification on colorized image. We first colorized the images from validation set of ImageNet100. As the size of dataset is large, we did not upload the images to repo. Images can be downloaded from https://www.kaggle.com/datasets/ambityga/imagenet100. Using convert2eccv.py, images can be colorized. In the original codes, the model uses T=0.38, and we cannot change the value of temperature from demo_release.py. In order to change the value, we look into the caffe branch of original repo. Then, we modify the codes in colorizers/eccv16.py, which it now accepts an argument `temperature`. The original file is put in colorizers/old/eccv16.py.

In colorizers/eccv.py, line 16 `colorizer_eccv16 = eccv16(temperature=0.3).eval()` allow us to change the temperature of the model. It is also the improvement mentioned in our report.

We use color2gray.py to transform color images to grayscale images. They are also used in image classification.

The file classification.ipynb perform image classification with VGG16. We test the accuracy with different images sets (original, grayscale, colorized). The top 5 predicted class are generated for each image, and the labels are output to the folder Prediction. Accuracy is calculated using Top-1 class.
