<div align="center">

Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time
=============================
Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, 
Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, 
Yair Carmon, Simon Kornblith, Ludwig Schmidt

arXiv 2022

<a href="data/model_soups.pdf"><img src="../../data/badge/paper_paper.svg"></a>
<a href="https://github.com/Burf/ModelSoups"><img src="../../data/badge/paper_code.svg"></a>
<a href="https://medium.com/@sabrinaherbst/model-soups-for-higher-performing-models-1d4818126191"><img src="../../data/badge/paper_reference.svg"></a>
</div>


## Highlight
Recently, researchers from various universities and companies published a 
paper on ArXiv, introducing a concept called **Model Soups**. 

Model Soups are supposed to boost the accuracy of machine learning models by 
averaging the weights of models trained with various hyperparameters.


## Method
<details open>
<summary><b style="font-size:18px">Current Training Workflow</b></summary>

Usually, after having picked out a model, data scientists will start to train 
the model with different hyperparameters and figure out which configurations 
work best. A lot of times the best configuration is determined either by a 
simple train-test split or a more rigorous Cross-Validation.

This one configuration will then be used while all other configurations 
(which might only be slightly worse than the best one) are discarded.
</details>

<br>
<details open>
<summary><b style="font-size:18px">Model Soups</b></summary>

> Model Soups allow averaging the weights of the models without requiring 
> additional memory or inference time. This is an advantage when compared to 
> ensemble methods. Still, Model Soups increase the robustness as ensemble 
> methods do.

Wortsman et al. [1] came up with three different approaches to so-called 
Model Soups: **Uniform Soups**, **Greedy Soups**, **Learned Soups**.
</details>


## Ablation Studies


## Results


## Citation
