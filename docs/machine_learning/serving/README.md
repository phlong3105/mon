---
layout      : default
title       : Model Serving
parent		: Machine Learning
nav_order   : 3
has_children: true
has_toc     : true
---

![data/model_deployment.png](data/model_deployment.png)

# Model Serving

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

A common grumble among data science or machine learning researchers or
practitioners is that putting a model in production is difficult. As a result,
some claim that a large percentage, 87%, of models never see the light of the
day in production.

However, there are some acceptable and common technical considerations and
pitfalls to keep in mind when considering your ML stack and tools. We’ll discuss
some common considerations and common pitfalls for tooling and best practices
and ML model serving patterns that are an essential part of your journey from
model development to deployment in production.

## Deploying, Serving and Inferencing Models at Scale

Once the model is trained and tested, with confidence that it met the
business requirements for model accuracy, seven crucial requirements for
scalable model serving frameworks to consider are:

- **Framework-agnostic**: A model serving-elected framework should be ML
  framework-agnostic. That is, it can deploy any common model built with common
  ML frameworks. For example, PyTorch, TensorFlow, XGBoost, or Scikit-learn,
  each with its own algorithms and model architectures.


- **Model Replication**: Some models are compute-intensive or network-bound.
  As such the elected framework can fan out requests over to model replicas,
  load balancing among replicas to support parallel request handling during peak
  traffic.


- **Request Batching**: Not all models in production are employed for
  real-time serving. Often, models are scored in large batches of requests.
  For example, for deep learning models, parallelize these image requests to
  multiple cores, taking advantage of hardware accelerators, to expedite batch
  scoring and utilize hardware resources is worthy of consideration.


- **High Concurrency and Low Latency**: Models in production require real-time
  inference with low latency while handling bursts of heavy traffic of requests.
  The consideration is crucial for best user experience to receive millisecond
  responses on prediction requests.


- **Model Deployment CLI and APIs**: A ML engineer responsible for deploying
  a model should be able to use model server’s deployment APIs or command line
  interfaces (CLI) simply to deploy model artifacts into production. This allows
  model deployment from within an existing CI/CD pipeline or workflow.


- **Patterns of Models in Production**: As ML applications are increasingly
  becoming pervasive in all sectors of industry, models trained for these ML
  applications are complex and composite. They range from computer vision to
  natural language processing to recommendation systems and reinforcement
  learning.

That is, models don’t exist in isolation. Nor do they predict results
singularly. Instead, they operate jointly and often in four model patterns:
**pipeline**, [**ensemble**](ensemble.md), **business logic**, and
**online learning**. Each pattern has its purpose and merit.

| ![data/ml_model_patterns_in_production.png](data/ml_model_patterns_in_production.png) |
|:-------------------------------------------------------------------------------------:|
|                                   <img width=600/>                                    |

## Observing and Monitoring Model in Production

Model monitoring, often an overlooked stage as part of model development
lifecycle, is critical to model’s viability in the post deployment production
stage. It is often an afterthought, at an ML engineer’s peril.

Why consider model monitoring? For a number of practical reasons, this stage is
pivotal. Let’s briefly discuss them.

- **Data drifts over time**: our quality and accuracy of the model depends on
  the quality of the data. Data is complex and never static, meaning what the
  original model was trained with the extracted features may not be as important
  over time. Some new features may emerge that need to be taken into account.
  For example, seasonal data changes. Such features drifts in data require
  retraining and redeploying the model, because the distribution of the
  variables is no longer relevant.


- **Model concept changes over time**: Many practitioners refer to this as
  model decay or model staleness. When the patterns of trained models no longer
  hold with the drifting data, the model is no longer valid because the
  relationships of its input features may not necessarily produce the model’s
  expected prediction. Hence, its accuracy degrades.


- **Models fail over time**: Models fail for inexplicable reasons: a system
  failure or bad network connection; an overloaded system; a bad input or
  corrupted request. Detecting these failures’ root causes early or its
  frequency
  mitigates user bad experience or deters mistrust in the service if the user
  receives wrong or bogus outcomes.


- **Systems degrade over load**: Constantly being vigilant of the health of
  your dedicated model servers or services deployed is just as important as
  monitoring the health of your data pipelines that transform data or your
  entire
  data infrastructure’s key components: data stores, web servers, routers,
  cluster nodes’ system health, etc.

## Methods

| Status | Method                          | Pattern                     | Date       | Publication    |
|:------:|---------------------------------|-----------------------------|------------|----------------|
|   ✅    | [**ModelSoups**](modelsoups.md) | [**Ensemble**](ensemble.md) | 2021/02/02 | AAAI&nbsp;2021 |
