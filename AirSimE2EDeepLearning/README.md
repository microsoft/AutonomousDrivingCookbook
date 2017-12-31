# Autonomous Driving using End-to-End Deep Learning: an AirSim tutorial

### Authors:

**[Mitchell Spryn](https://www.linkedin.com/in/mitchell-spryn-57834545/)**, Software Engineer II, Microsoft

**[Aditya Sharma](https://www.linkedin.com/in/adityasharmacmu/)**, Program Manager, Microsoft

## Overview

In this tutorial, you will learn how to train and test an end-to-end deep learning model for autonomous driving using data collected from the [AirSim simulation environment](https://github.com/Microsoft/AirSim). You will train a model to learn how to steer a car through a portion of the Landscape map in AirSim using only one of the front facing webcams as visual input. Such a task is usually considered the "hello world" of autonomous driving, but after finishing this tutorial you will have enough background to start exploring new ideas on your own. Through the length of this tutorial, you will also learn some practical aspects and nuances of working with end-to-end deep learning methods.

Here's a short sample of the model in action:

![car-driving](car_driving.gif)



## Structure of this tutorial

The code presented in this tutorial is written in [Keras](https://keras.io/), a high-level deep learning Python API capable of running on top of [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/), [TensorFlow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/index.html). The fact that Keras lets you work with the deep learning framework of your choice, along with its simplicity of use, makes it an ideal choice for beginners, eliminating the learning curve that comes with most popular frameworks.

This tutorial is presented to you in the form of Python notebooks. Python notebooks make it easy for you to read instructions and explanations, and write and run code in the same file, all with the comfort of working in your browser window. You will go through the following notebooks in order:

**[DataExplorationAndPreparation](DataExplorationAndPreparation.ipynb)**

**[TrainModel](TrainModel.ipynb)**

**[TestModel](TestModel.ipynb)**

If you have never worked with Python notebooks before, we highly recommend [checking out the documentation](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).

## Prerequisites and setup

#### Background needed

At the very least, you need to be familiar with the basics neural networks. You are not required to know advanced concepts like LSTMs or Reinforcement Learning but you should know how Convolutional Networks work. A really good starting point to get a strong background in a very short amount of time is [this great book](http://neuralnetworksanddeeplearning.com/) written by Michael Nielsen. It is free, very short and available online. It can provide you a solid foundation in less than a week's time.

You should also be comfortable with Python. At the very least, you should be able to read and understand code written in Python. 

#### Environment Setup

1. [Install AirSim](https://github.com/Microsoft/AirSim#how-to-get-it)
2. [Install Anaconda](https://conda.io/docs/user-guide/install/index.html) with Python 3.5 or higher.
3. [Install CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine) or [install Tensorflow](https://www.tensorflow.org/install/install_windows)
4. [Install h5py](http://docs.h5py.org/en/latest/build.html)
5. [Install Keras](https://keras.io/#installation)
6. [Configure Keras backend](https://keras.io/backend/) to work with TensorFlow (default) or CNTK.

#### Hardware

It is highly recommended that a GPU is available for processing. While it is possible to train the model using just a CPU, it will take days to complete training. This tutorial was developed with a Nvidia GTX970 GPU, which resulted in a training time of ~45 minutes. 

If you do not have a GPU available, you can spin up a [Deep Learning VM on Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning), which comes with all the dependencies and libraries installed (use the provided py35 environment if you are using this VM).

#### Dataset

The dataset for the model is quite large. [You can download it from here](https://aka.ms/AirSimTutorialDataset). The first notebook will provide guidance on how to access the data once you have downloaded it. The final uncompressed data set size is approximately 3.25GB (which although is nothing compared to the petabytes of data needed to train an actual self-driving car, should be enough for the purpose of this tutorial).

#### A note from the curators

We have made our best effort to ensure this tutorial can help you get started with the basics of autonomous driving and get you to the point where you can start exploring new ideas independently. We would love to hear your feedback on how we can improve and evolve this tutorial. We would also love to know what other tutorials we can provide you that will help you advance your career goals. Please feel free to use the GitHub issues section for all feedback. All feedback will be monitored closely. If you have ideas you would like to [collaborate](../README.md#contributing) on, please feel free to reach out to us and we will be happy to work with you.