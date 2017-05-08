
# SegNet

This is a model implemented as part of my master thesis, based on the SegNet paper. The goal is to use the model to segment multispectral images, so that building information can be extracted.


## Tensorflow
The network is implemented by using the library TensorFlow.

## Recognition
The code is based on the SegNet implementation by tkuanlan35: https://github.com/tkuanlun350/Tensorflow-SegNet. Updates done are: 
- Upgraded to TensorFlow 1.0.1
- Added dropout layers
- Added method for unraveling indices so that the max-pooling indices can be used in the decoder to upsample the images 
- Added flags

## Requirements:
- Tensorflow GPU 1.0.1
- Python 3.5

See more in requirements.txt.

## Usage
#### Requirements
pip install -r requirements.txt

#### Run TensorBoard:
tensorboard --logdir=path/to/log-directory

#### Dataset
To verify the model I used the CamVid dataset. This can be downloaded from: https://github.com/alexgkendall/SegNet-Tutorial, and used in the model by setting the correct paths and dataset size in model_train.py.

The Aerial dataset I use is unfortunately not open yet. 
