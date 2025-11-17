# Automotive object detection via a SSD model built from scratch
In this project, the single shot multibox detector (SSD) model of W. Liu, et al. (see [here](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)) is constructed from scratch.
This model is trained and tested on an automotive focused dataset, but all ideas and code are applicable in other scenarios.

## Model Architecture
SSD is a feed-forward convolutional neural network.  The early layers of the model are the well known VGG-16 network (see [here](https://ieeexplore.ieee.org/document/7486599)).

![description](/figures/SSD_architecture.png)

In the above figure (from the paper of W. Liu, et al.), we can see the architecture of the SSD model.
Many advances have been made since the introduction of SSD.
Two such improvements have been implemented in the model here.  First, batch norms now follow most of the convolution layers, and, second, intersection over union (IoU) based non-maximum supression is upgraded to complete IoU based non-maximum supression.
The constuction in this project (see [here](/SSD_from_scratch.py)) is not elegant, as the intent is to show model architecture for educational purposes.  A cleaner implementation can be found [here](https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py).
