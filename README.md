# SSLearningPipeLine

This project is a semi supervised pipeline for locating objects in images. Presently the pipeline is transfer learning to find a fixed set of boxes in an image.

for a fixed set of makes use of transfer learning. A small set of images will

Users are expected to write a script that prepares images to label.
This script imports this package to 

  1. run the labeling tool
  2. managing the labels
  3. produce 
 

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

