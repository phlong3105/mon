---
layout      : default
title       : Out-of-Distribution Generalization
parent        : Model Learning
grand_parent: Machine Learning
has_children: false
has_toc     : false
permalink   : /machine_learning/model_learning/ood
---

# Out-of-Distribution Generalization

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

Classic machine learning methods are built on the assumption that training and
testing data are **independent and identically distributed (IID)**. However, in
real scenarios, deep neural networks struggle and lack robustness when the test
data are in different distributions from the training data. These are called 
**out-of-distribution(OoD)** problems.

The main challenge in OoD generalization is the **domain shift** between source
and target distributions and the inaccessibility of the target data. Data in the
target domain is never seen during the training time, which leads to
overfitting on the source domains and poor performance in target domains.

##           
