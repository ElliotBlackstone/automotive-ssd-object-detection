# Automotive object detection via a SSD model built from scratch
In this project, the single shot multibox detector (SSD) model of W. Liu, et al. (see [here](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)) is constructed from scratch.
This model is trained and tested on an automotive focused dataset, but all ideas and code are applicable in other scenarios.


## Model Architecture
SSD is a feed-forward convolutional neural network.  The early layers of the model are the well known VGG-16 network (see [here](https://ieeexplore.ieee.org/document/7486599)).

![SSD model architecture by W. Liu, et al.](/figures/SSD_architecture.png)

In the above figure (from the paper of W. Liu, et al.), we can see the architecture of the SSD model.
Many advances have been made since the introduction of SSD.
Two such improvements have been implemented in the model here.  First, batch norms now follow most of the convolution layers, and, second, intersection over union (IoU) based non-maximum supression is upgraded to complete IoU based non-maximum supression.
The constuction in this project (see [here](/SSD_from_scratch.py)) is not elegant, as the intent is to show model architecture for educational purposes.  A cleaner implementation by Max deGroot can be found [here](https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py).


## Dataset
The [dataset](https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset), originally compiled by Udacity, contains 29,800 images with 194,539 bounding boxes with classification labels.  Two such examples are below.

![An image with ground truth bounding boxes and classification.](/figures/idx_6932_GT_Box_Label.png)
![An image with ground truth bounding boxes and classification.](/figures/idx_2265_GT_Box_Label.png)

Unfortunately, there are some flaws with the dataset, as seen below.

![An image with ground truth bounding boxes and classification.](/figures/idx_1568_GT_Box_Label.png)
![An image with ground truth bounding boxes and classification.](/figures/idx_5979_bad_target_example.png)

In the first image, two of the traffic lights are labels twice and in the second image, a house is labeled as a truck.  These labeling mistakes are present throughout the dataset but occur infrequently.


## Preprocessing

After downloading the [dataset](https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset), follow the steps in the [preprocessing_car.ipynb](/preprocessing_car.ipynb) notebook.


Originally, there is a total of 11 classes with 7 classes being related to traffic lights (green, red, left turn, etc.).  For simplicity, all of the traffic light classes have been grouped together, which gives us 5 classes (biker, car, pedestrian, traffic light, truck).
Lastly, there are 3,500 'background' images (i.e. images containing no biker, car, pedestrian, traffic light, truck).

A train test split is created via a group stratified split.  Groups are images and stratification is with respect to class labels.




## Model training

Due to hardware constraints, only 20% of the training dataset was used to train the model.  
The model was trained 103 epochs using the SGD optimizer and a ReduceLROnPlateau scheduler.  The number of epochs was a result of early stopping rounds (model training was halted to prevent overfitting).


![Training loss data](/figures/loss_vs_epoch.png)













