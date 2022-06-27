---
layout      : default
title       : Model Learning
parent		: Machine Learning
nav_order   : 2
has_children: true
has_toc     : false
permalink   : /machine_learning/model_learning
---

![data/training.png](data/training.png)

# Model Learning

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

Neural networks learn a mapping function from inputs to outputs that can be
summarized as solving the problem of function approximation.

Unlike other machine learning algorithms, the parameters of a neural network
must be found by solving a non-convex optimization problem with many good
solutions and many misleadingly good solutions.

This is achieved by updating the weights of the network in response to the
errors the model makes on the training dataset. Updates are made to continually
reduce this error until either a good enough model is found or the learning
process gets stuck and stops. The stochastic gradient descent algorithm is used
to solve the optimization problem where model parameters are updated each
iteration using the backpropagation algorithm.

The process of training neural networks is the most challenging part of using
the technique in general and is by far the most time-consuming, both in terms
of effort required to configure the process and computational complexity
required to execute the process.

## Neural Nets Learn a Mapping Function

Deep learning neural networks learn a mapping function.

- Developing a model requires historical data from the domain that is used as
  training data.
- This data is comprised of observations or examples from the domain with
  input elements that describe the conditions and an output element that
  captures what the observation means.

We can describe the relationship between the input variables and the output
variables as a complex mathematical function.

> A feedforward network defines a mapping and learns the value of the parameters
> that result in the best function approximation. As such, we can describe the
> broader problem that neural networks solve as “function approximation.”

They learn to approximate an unknown underlying mapping function given a
training dataset. They do this by learning weights and the model parameters,
given a specific network structure that we design.

A neural network model must learn in such a way that mapping works well for
the training dataset, but also works well on new examples not seen by the model
during training. This ability to work well on specific examples and new examples
is called the ability of the model to [**generalize**](generalization.md).

## Learning Network Weights Is Hard

> Finding the parameters for neural networks in general is hard. In fact,
> training a neural network is the most challenging part of using the
> technique.

For many simple machine learning algorithms, finding parameters for many
machine learning algorithms involves solving a **convex optimization problem**:
that is an error surface that is shaped like a bowl with a single best solution.

- For example, we can use linear algebra to calculate the specific
  coefficients of a linear regression model and a training dataset that best
  minimizes the squared error.
- Similarly, we can use optimization algorithms that offer convergence
  guarantees when finding an optimal set of model parameters for nonlinear
  algorithms such as logistic regression or support vector machines.

For deep learning neural networks, we can neither directly compute the optimal
set of weights for a model, nor can we get global convergence guarantees to
find an optimal set of weights.

- The use of nonlinear activation functions in the neural network means that
  the optimization problem that we must solve in order to find model parameters
  **is not convex**.
- It is not a simple bowl shape with a single best set of weights that we are
  guaranteed to find. Instead, there is a landscape of peaks and valleys with
  many good and many misleadingly good sets of parameters that we may discover.

> Solving this optimization is challenging, not least because the error surface
> contains many local optima, flat spots, and cliffs.

An iterative process must be used to navigate the non-convex error surface of
the model. A naive algorithm that navigates the error is likely to become
misled, lost, and ultimately stuck, resulting in a poorly performing model.

## Navigating the Non-Convex Error Surface

Neural network models can be thought to learn by navigating a **non-convex
error surface**.

- A model with a specific set of weights can be evaluated on the training
  dataset and the average error over all training datasets can be thought of as
  the error of the model.
- A change to the model weights will result in a change to the model error.
  Therefore, we seek a set of weights that result in a model with a small error.
- This involves repeating the steps of evaluating the model and updating the
  model parameters in order to step down the error surface. This process is
  repeated until a set of parameters is found that is good enough or the search
  process gets stuck.

**Optimization**: is a search or an optimization process that operates in this
way as gradient optimization algorithms, as they naively follow along the error
gradient. They are computationally expensive, slow, and their empirical behavior
means that using them in practice is more art than science.

> The algorithm that is most commonly used to navigate the error surface is
> called **stochastic gradient descent (SGD)**.

Other global optimization algorithms designed for non-convex optimization
problems could be used, such as a genetic algorithm, but SGD is more efficient
as it uses the gradient information specifically to update the model weights
via an algorithm called [**backpropagation**](backproppagation.md).

**Backpropagation** refers to a technique from calculus to calculate the
derivative (e.g. the slope or the gradient) of the model error for specific
model parameters, allowing model weights to be updated to move down the
gradient. As such, the algorithm used to train neural networks is also often
referred to as simply backpropagation.

## Components of the Learning Algorithm

Training a deep learning neural network model using SGD with backpropagation
involves choosing a number of components and hyperparameters.

- **Loss Function**:
  - An error function must be chosen, often called the objective function,
    cost function, or the loss function.
  - The function used to estimate the performance of a model with a specific
    set of weights on examples from the training dataset.


- **Weight Initialization**:
  - The search or optimization process requires a starting point from which to
    begin model updates. The starting point is defined by the initial model
    parameters or weights.
  - Because the error surface is non-convex, the optimization algorithm is
    sensitive to the initial starting point.
  - As such, small random values are chosen as the initial model weights,
    although different techniques can be used to select the scale and
    distribution of these values.


- **Batch Size**:
  - The number of examples used to estimate the error gradient before updating
    the model parameters.


- **Learning Rate**:
  - The amount that each model parameter is updated per cycle of the learning
    algorithm.
  - Controls how much to update model weights and, in turn, controls how fast a
    model learns on the training dataset.


- **Epochs**:
  - The number of complete passes through the training dataset before the
    training process is terminated.
  - The training process must be repeated many times until a good or good
    enough set of model parameters is discovered.
  - The total number of iterations of the process is bounded by the number of
    complete passes through the training dataset after which the training
    process is terminated.

## Learning Types

| Type                                                                    | Description                                                                                                                                          |
|-------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [**N&#8209;Shot&nbsp;Learning&nbsp;(NSL)**]()                           | N-Shot Learning is seen as a more broad concept than all the others. It means that Few-Shot, One-Shot, and Zero-Shot Learning are sub-fields of NSL. |
| [**Zero&#8209;Shot&nbsp;Learning&nbsp;(ZSL)**]()                        | The goal of Zero-Shot Learning is to predict unseen classes without any training examples.                                                           |
| [**One&#8209;Shot&nbsp;Learning&nbsp;(OSL)**]()                         | One-Shot predicts unseen classes with only a single sample of each class.                                                                            |
| [**Few&#8209;Shot&nbsp;Learning&nbsp;(FSL)**](fsl.md)                   | Few-Shot predicts unseen classes with two to five samples per class, making it just a more flexible version of OSL.                                  | 
| [**Out&#8209;of&#8209;Distribution&nbsp;(OoD) Generalization**](ood.md) |                                                                                                                                                      |

## Methods

| Status | Method                  | Technique                                                                    | Date       | Publication |
|:------:|-------------------------|------------------------------------------------------------------------------|------------|-------------|
|   📑   | [**DecAug**](decaug.md) | [**Out&#8209;of&#8209;Distribution&nbsp;(OoD)&nbsp;Generalization**](ood.md) | 2021/05/10 | arXiv       |
