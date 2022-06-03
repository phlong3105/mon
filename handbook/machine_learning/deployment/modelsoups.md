<div align="center">
<br><br>
<div>
	<a href="https://github.com/phlong3105/one/blob/master/handbook/README.md"><img src="../../data/badge/handbook_home.svg"></a>
	<a href="https://github.com/phlong3105/one/blob/master/handbook/machine_learning/README.md"><img src="../../data/badge/handbook_machine_learning.svg"></a>
	<a href="https://github.com/phlong3105/one/blob/master/handbook/machine_learning/deployment/README.md"><img src="../../data/badge/handbook_deployment.svg"></a>
</div>

Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time
=============================
<div>
	<p>Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, 
       Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, 
	   Yair Carmon, Simon Kornblith, Ludwig Schmidt</p>
	<p>arXiv 2022</p>
</div>

<div align="center">
	<a href="data/model_soups.pdf"><img src="../../data/badge/paper_paper.svg"></a>
	<a href="https://github.com/Burf/ModelSoups"><img src="../../data/badge/paper_code.svg"></a>
    <a href="https://medium.com/@sabrinaherbst/model-soups-for-higher-performing-models-1d4818126191"><img src="../../data/badge/paper_reference.svg"></a>
</div>
</div>


## Highlight
- Recently, researchers from various universities and companies published a 
paper on ArXiv, introducing a concept called **Model Soups**. 
- Model Soups are supposed to boost the accuracy of machine learning models by 
averaging the weights of models trained with various hyperparameters.


## Method
<details open>
<summary><b style="font-size:18px">Current Training Workflow</b></summary>

- Usually, after having picked out a model, data scientists will start to train 
the model with different hyperparameters and figure out which configurations 
work best. A lot of times the best configuration is determined either by a 
simple train-test split or a more rigorous Cross-Validation.


- This one configuration will then be used while all other configurations 
(which might only be slightly worse than the best one) are discarded.
</details>


<details open>
<summary><b style="font-size:18px">Model Soups</b></summary>


</details>

## Ablation Studies


## Results


## Citation
