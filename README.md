# SSLearningPipeLine

Semi supervised pipeline for locating objects in images -- intended for detector images read from LCLS experiments with psana. The user labels a fixed number of objects in a couple hundred images. The pipeline builds a model to predict the fixed objects in new images. The pipeline is

1. user writes functions to get and prepare images
   1. uses psana to get detector images from experiment data 
   1. calibrates images, subtracts dark if neccessary
   1. extracts ROI, applies normalizations, like log transform
1. user initializes instance of pipeline
   1. choose total number of images to label, call this N
   1. decides how many distinct objects can be present in an image
1. randomly goes through images
1. associates unique key with each image (use psana event id's)
1. calls pipeline to label until N images labeled
1. calls pipeline build model function
1. loops through new images
1. calls pipeline predict function

Presently the pipeline is not for finding all of something, i.e, all braggs peaks in diffraction images. It is for finding a few distinct objects. For instance, in a two color experiment, there are two things that might be present, high energy lasing and low energy lasing -- and each will occur 1 or 0 times in the images. 

On the pipeline side, when label is called,

1. first shows the image back to the user, asks if user wants to label, user may not want to label because
   1. not all shots may have the signal of interest
   1. a sufficient number of samples of one category is already labeled - i.e, we may have many labels of both high and low energy lasing, but very few of just high energy lasing.
1. brings up external pylabelme program to label
   1. presently, we have hacked tool to make boxes
   1. click polygon
   1. make two points, opposite corners of box
   1. make the name 0, 1, .. up to M-1 where M is the number of distinct things that might be present
   1. be consistant about what you name 0, 1, etc
   1. click save when done
1. calls vgg16 on the image, saves the feature vector

After labeling is done, when model building starts

1. If there are M boxes, builds a classifier for 2^M categories
1. for each of the 2^M categories, builds a separate linear regressor to predict the specific boxes present

It is important to label enough samples from each of the separate 2^M categories.
 

# Operations

You need to have both this repository, and davidslac/pylabelme checked out. Then before you run, you need to adjust 
your PYTHONPATH so that when you run the SSLearningPipeline user_driver.py, it can find the labelme tool.

Suggestion,

create a working directory, ie

```
mkdir work
cd work
```

source /reg/g/psdm/bin/conda_setup
on pslogin (outside internet machine, get both these repos):

```
git clone https://github.com/mmongia/SSLearningPipeLine.git
git clone https://github.com/davidslac/pylabelme.git
```

now in another terminal, 

```
ssh psana
source /reg/g/psdm/bin/conda_setup
cd work/SSLearningPipeline
pyqtcrc command for resources

PYTHONPATH=../pylabelme:$PYTHONPATH python user_driver.py
```

notice that the script, user_driver.py, is telling sslearn to write the labeled files into 
```
/reg/d/psdm/amo/amo86815/scratch/davidsch
```
to get going, make your own directory in scratch, edit user_driver.py for yourself.

