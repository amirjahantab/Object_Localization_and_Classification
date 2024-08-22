# Object Localization and Classification

This repository contains the implementation of an object localization and classification model, focusing on the task of identifying and locating objects within images. The project leverages the power of deep learning, specifically using the ResNet34 architecture, to perform both classification and localization tasks simultaneously. 



## Dataset

The dataset used in this project is from the [Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) competition on Kaggle. The dataset consists of images of various fish species, captured under different conditions. The primary goal is to classify the species of fish and localize them within the images.


## Model Architecture

The model is built using the ResNet34 architecture, a deep convolutional neural network known for its performance on image classification tasks. This model is adapted to perform both object classification and localization.

### ResNet34

ResNet34 is a type of residual network introduced in the paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) by Kaiming He et al. It is part of the ResNet family, known for introducing residual learning, which helps in training very deep networks by mitigating the vanishing gradient problem.




### Combined Model for Classification and Localization

The model's architecture consists of the following components:

1. **Base Model**: ResNet34 is used as the base model for feature extraction.
2. **Classification Head**: A fully connected layer that outputs the class probabilities.
3. **Localization Head**: Another fully connected layer that outputs the bounding box coordinates (x, y, width, height).

The model is trained using a combined loss function that accounts for both classification accuracy and the precision of the bounding box predictions.



## Accuracy
$$ Accuracy = 95 \% $$