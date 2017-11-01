---
title:  "ML Nano Degree - Capstone Project Proposal"
date:   2017-10-30 20:00:00
categories: [machine learning]
tags: [machine learning]
---


## 1. Introduction
Deep learning with Deep Neural Network has gained huge traction in the past few years owning to the blooming of data and GPU accelerated computation. The former makes it possible to collect massive amount of training data to tune millions or even billions of parameters; the later makes running deep learning computation approachable for everyone with consumer level GPU installed personal PC. 

One of the most exciting field of deep learning application is computer vision which gives machine the ability to see. One of the most influential innovations in the field of computer vision is Convolution Neurual Network(CNN), first published by LeCun et al [1], became prominence as Alex Krizhevsky used it to win 2012 ImageNet competition. The key idea of CNN is weight sharing unlike Multilayer Perceptrons (MLP) where each input has its own weight. This allows CNN to extract patterns from images regardless of their position. First layer extracts simple patters such as edges. Following layers extract more abstract patterns based on the previous layers [2]. 

![Visualize Layers]({{ site.url }}/assets/mlnano_capstone_visualize_layers.png) 

Though empowered by CNN and GPU, training an image classifier still requires massive amount of data for each class and a long computation time which is often not feasible for individuals or not effective enough to test out new ideas. As aforementioned, the first few layers extract relatively simple patterns which are roughly the same for different image classification tasks. For instance, basic patterns like edges and curves are the basic elements of every image [3] (Lee et al 2009). 

![Learning Objects Parts]({{ site.url }}/assets/mlnano_capstone_learning_object_parts.png)

Transfer learning come to the rescue. Transfer learning allows us to reuse the less abstract layers of a trained model for a similar tasks and achieve a different goal by training the top layers with way less data and effort due to the massive reduce of trainable parameters. 

In this study, we look to train an animal image classifier with 19 classes. 

## 2. Project Plan 

### 2.1 Dataset
The KTH-ANIMALS dataset[4] contains only 1740 images in total and are of different shapes from ~200 to ~400 pixels width and height. We first load them into the same shape 224 by 224 pixels. Plot them to validate if class labels have been aligned correctly. 

![Input Images]({{ site.url }}/assets/mlnano_capstone_visualize_input_images.png)

As can been seen in the training data, the images are taken from different angles in the wild. This gives us even more troubles. For example, higher model complexity is required to learn goat from the front as well as from the side when the goat horn is probably used by the model to identify goat.

We also have a look at the label count to make sure the dataset is not skewed.
![Input Images]({{ site.url }}/assets/mlnano_capstone_training_label_count.png)

We then cut the data into training, validation and test set containing 1460, 180 and 100 images respectively. Leaving us merely 78 images on average per animal class which is far from sufficient to train a CNN model from stratch to classify such a diverse dataset. 

### 2.2 Benchmark Model
We then run our benchmark model - a CNN model with at least 5 ConvNet layers. Measure the classification accuracy. 

### 2.3 Project Design
We run a transfer learning model - a VGG19 with weights pretrained on ImageNet dataset with top layers replaced. Train the weight of the top model with the same data. Measure the classification accuracy. 
We then freeze the first 4 CNN blocks of VGG19 to fine tune the last(5th) block of VGG19 as well as our top model and compare the classification accuracy with our benchmark model. The transfer learning model should show a significant increase in accuracy. 

At the end we will also explore utilizing the last CNN output weights for localization. 

... To Be Finished ...







## Reference
[1] http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf <br/>
[2] https://arxiv.org/pdf/1311.2901.pdf <br/>
[3] https://www.cs.princeton.edu/~rajeshr/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf <br/>
[4] https://www.csc.kth.se/~heydarma/Datasets.html
[5] https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

