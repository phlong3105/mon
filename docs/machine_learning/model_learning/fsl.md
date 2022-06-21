---
layout      : default
title       : Few-Shot Learning
parent      : Model Learning
grand_parent: Machine Learning
has_children: false
has_toc     : false
permalink   : /machine_learning/model_learning/fsl
---

# Few-Shot Learning

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

_---

**Few-Shot Learning** (FSL) is a sub-area of machine learning. It’s about
classifying new data when you have only a few training samples with supervised
information.

When we’re talking about FSL, we usually mean **N-way-K-Shot Learning**, where
**N** stands for the number of classes, and **K** for the number of samples from
each class to train on.

## Few-Shot Learning Approaches

First, let’s define an **N-way-K-Shot Learning** problem. Imagine that we have:

* A training (support) set that consists of:
	* **N** class labels
	* **K** labeled images for each class (a small amount, less than ten samples
	  per class)
* **Q** query images

* We want to classify **Q** query images among the **N** classes. The **N * K**
  samples in the training set are the only examples that we have. The main
  problem here is not enough training data.

Generally, there are two approaches that you should consider when solving FSL
problems:

* Data-level approach (DLA)
* Parameter-level approach (PLA)

### Data-level Approach

This approach is really simple. It’s based on the concept that if you don’t have
enough data to build a reliable model and avoid overfitting and underfitting,
you should **simply add more data**.

Method 1: use additional information from a larger base-dataset. **The key
feature of the base-dataset is that it doesn't have classes that we have in our
support-set for the Few-Shot task**. For example, if we want to classify a
specific bird species, the base-dataset can have images of many other birds.

Method 2: produce more data. To reach this goal, we can use data augmentation,
synthetic data, or even generative adversarial networks (GANs).

### Parameter-level Approach

From the parameter-level point of view, it’s quite **easy to overfit on Few-Shot
Learning samples**, as they have extensive and high-dimensional spaces quite
often.

To overcome this problem we should limit the parameter space and use
regularization and proper loss functions. The model will generalize the limited
number of training samples.

On the other hand, we can enhance model performance by directing it to the
extensive parameter space. If we use a standard optimization algorithm, it might
not give reliable results because of the small amount of training data.

That is why on the parameter-level we train our model to find the best route in
the parameter space to give optimal prediction results. This technique is
called **Meta-Learning**.
