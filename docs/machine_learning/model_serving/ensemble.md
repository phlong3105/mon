---
layout      : default
title       : Ensemble
parent	    : Model Serving
grand_parent: Machine Learning
nav_order   : 1
has_children: false
has_toc     : false
permalink   : /machine_learning/model_serving/ensemble
---

![data/ensemble.png](data/ensemble.png)

# Ensemble Models

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

Ensemble models is a machine learning approach to combine multiple other models
in the prediction process. Those models are referred to as base estimators. It
is a solution to overcome the following technical challenges of building a
single estimator:

- High variance: The model is very sensitive to the provided inputs to the
  learned features.
- Low accuracy: One model or one algorithm to fit the entire training data might
  not be good enough to meet expectations.
- Features noise and bias: The model relies heavily on one or a few features
  while making a prediction.

## Ensemble Algorithms

A single algorithm may not make the perfect prediction for a given dataset.
Machine learning algorithms have their limitations and producing a model with
high accuracy is challenging.

If we build and combine multiple models, the overall accuracy could get boosted.
The combination can be implemented by aggregating the output from each model
with two objectives: reducing the model error and maintaining its
generalization.

The way to implement such aggregation can be achieved using some techniques.
Some textbooks refer to such architecture as **meta-algorithms**.

| ![data/ensemble_algorithms.png](data/ensemble_algorithms.png) |
|:-------------------------------------------------------------:|
| Diversifying the model predictions using multiple algorithms. |
|                      <img width="600"/>                       |

## Ensemble Learning

Ensemble learning is a combination of several machine learning models in one
problem. These models are known as **weak learners**. The intuition is that when
you combine several weak learners, they can become **strong learners**.

Each weak learner is fitted on the training set and provides predictions
obtained. The final prediction result is computed by combining the results from
all the weak learners.

One important aspect to mention is those weak learners can have different ways
of mapping the features with variant decision boundaries.

|         ![data/ensemble_learning.png](data/ensemble_learning.png)          |
|:--------------------------------------------------------------------------:|
| Aggregated predictions using multiple weak learners of the same algorithm. |
|                             <img width="600"/>                             |

## Basic Ensemble Techniques

### Max Voting

The max voting method is generally used for classification problems. In this
technique, multiple models are used to make predictions for each data point. The
predictions by each model are considered as a `vote`. The predictions which we
get from the majority of the models are used as the final prediction.

For example, when you asked 5 of your colleagues to rate your movie (out of 5);
we’ll assume three of them rated it as 4 while two of them gave it a 5. Since
the majority gave a rating of 4, the final rating will be taken as 4. You can
consider this as taking the mode of all the predictions. The result of max
voting would be something like this:

| Colleague 1 | Colleague 2 | Colleague 3 | Colleague 4 | Colleague 5 | Final Rating |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|      5      |      4      |      5      |      4      |      4      |    **4**     |

### Averaging

In this method, we take an average of predictions from all the models and use it
to make the final prediction. Averaging can be used for making predictions in
regression problems or while calculating probabilities for classification
problems.

For example, the averaging method would take the average of all the values.
i.e. (5+4+5+4+4)/5 = 4.4

| Colleague 1 | Colleague 2 | Colleague 3 | Colleague 4 | Colleague 5 | Final Rating |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|      5      |      4      |      5      |      4      |      4      |   **4.4**    |

### Weighted Average

All models are assigned different weights defining the importance of each model
for prediction. For instance, if two of your colleagues are critics, while
others have no prior experience in this field, then the answers by these two
friends are given more importance as compared to the other people.

The result is calculated as [(5*0.23) + (4*0.23) + (5*0.18) + (
4*0.18) + (4*0.18)] = 4.41:

|        | Colleague 1 | Colleague 2 | Colleague 3 | Colleague 4 | Colleague 5 | Final Rating |
|:-------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| weight |    0.23     |    0.23     |    0.18     |    0.18     |    0.18     |              |
| rating |      5      |      4      |      5      |      4      |      4      |   **4.41**   |

## Advanced Ensemble Techniques

### Stacking

Stacking is a learning technique that uses predictions from multiple models (for
example decision tree, knn or svm) to build a new model. This model is used for
making predictions on the test set.

Below is a step-wise explanation for a simple stacked ensemble:

| Step                                                                                                                                                              |                                      | 
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------|
| 1. The train set is split into 10 parts                                                                                                                           | ![Step 1](data/stacking_step_01.png) |
| 2. A base model (suppose a decision tree) is fitted on 9 parts and predictions <br> are made for the 10th part. This is done for each part of the train set.      | ![Step 2](data/stacking_step_02.png) |
| 3. The base model (in this case, decision tree) is then fitted on the whole train dataset. <br> 4. Using this model, predictions are made on the test set.        | ![Step 3](data/stacking_step_03.png) |
| 5. Steps 2 to 4 are repeated for another base model (say knn) resulting in another <br> set of predictions for the train set and test set.                        | ![Step 5](data/stacking_step_05.png) |
| 6. The predictions from the train set are used as features to build a new model. <br> 7. This model is used to make final predictions on the test prediction set. | ![Step 6](data/stacking_step_06.png) |

### Blending

Blending follows the same approach as **stacking** but uses only a holdout (
validation) set from the train set to make predictions. In other words, unlike
stacking, the predictions are made on the holdout set only. The holdout set and
the predictions are used to build a model which is run on the test set.

Here is a detailed explanation of the blending process:

| Step                                                                                                                                                                     |                                      | 
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------|
| 1. The train set is split into training and validation sets.                                                                                                             | ![Step 1](data/blending_step_01.png) |
| 2. Model(s) are fitted on the training set. <br> 3. The predictions are made on the validation set and the test set.                                                     | ![Step 2](data/blending_step_03.png) |
| 4. The validation set and its predictions are used as features to build a new model. <br> 5. This model is used to make final predictions on the test and meta-features. |                                      |

### Bagging

The idea behind bagging is combining the results of multiple models (for
instance, all decision trees) to get a generalized result. Here’s a question: If
you create all the models on the same set of data and combine it, will it be
useful? There is a high chance that these models will give the same result since
they are getting the same input. So how can we solve this problem? One of the
techniques is bootstrapping.

**Bootstrapping** is a sampling technique in which we create subsets of
observations from the original dataset, with replacement. The size of the
subsets is the same as the size of the original set.

| ![data/bootstrapping.png](data/bootstrapping.png) |
|:-------------------------------------------------:|
|                   Boostrapping.                   |
|                <img width="600"/>                 |

**Bagging** (or Bootstrap Aggregating) technique uses these subsets (bags) to
get a fair idea of the distribution (complete set). The size of subsets created
for bagging may be less than the original set.

1. Multiple subsets are created from the original dataset, selecting
   observations with replacement.
2. A base model (weak model) is created on each of these subsets.
3. The models run in parallel and are independent of each other.
4. The final predictions are determined by combining the predictions from all
   the models.

| ![data/bagging.png](data/bagging.png) |
|:-------------------------------------:|
|               Bagging.                |
|          <img width="400"/>           |

### Boosting

Before we go further, here’s another question for you: If a data point is
incorrectly predicted by the first model, and then the next (probably all
models), will combining the predictions provide better results? Such situations
are taken care of by boosting.

**Boosting** is a sequential process, where each subsequent model attempts to
correct the errors of the previous model. The succeeding models are dependent on
the previous model. 

> Thus, the boosting algorithm combines a number of weak learners to form a
> strong learner. The individual models would not perform well on the entire
> dataset, but they work well for some part of the dataset. Thus, each model
> actually boosts the performance of the ensemble.

Let’s understand the way boosting works in the below steps:

| Step                                                                                                                                                                                                                                                                                                                                                                                         |                                      | 
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------|
| 1. A subset is created from the original dataset. <br> 2. Initially, all data points are given equal weights. <br> 3. A base model is created on this subset. <br> 4. This model is used to make predictions on the whole dataset                                                                                                                                                            | ![Step 4](data/boosting_step_04.png) |
| 5. Errors are calculated using the actual values and predicted values. <br> 6. The observations which are incorrectly predicted, are given higher weights. <br> (Here, the three misclassified blue-plus points will be given higher weights) <br> 7. Another model is created and predictions are made on the dataset <br> (This model tries to correct the errors from the previous model) | ![Step 7](data/boosting_step_07.png) |
| 8. Similarly, multiple models are created, each correcting the errors of the previous model. <br> 9. The final model (strong learner) is the weighted mean of all the models (weak learners).                                                                                                                                                                                                | ![Step 9](data/boosting_step_09.png) |
