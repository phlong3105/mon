<div align="center">
<h1>Wakeup-Darkness</h1>
</div>

<div align="center">
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/zhangbaijin/Wakeup-Darkness?color=green"> <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/zhangbaijin/Wakeup-Darkness">  <img alt="GitHub issues" src="https://img.shields.io/github/issues/zhangbaijin/Wakeup-Darkness"> 
</div>
<div align="center">
<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/zhangbaijin/Wakeup-Darkness?style=social"> <img alt="GitHub stars" src="https://img.shields.io/github/stars/zhangbaijin/Awesome-LLM-explainable"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/zhangbaijin/Wakeup-Darkness?style=social">
</div>



# Wakeup-Darkness: When Multimodal Meets Unsupervised Low-light Image Enhancement
# The structure
![image](https://github.com/zhangbaijin/Wakeup-Darkness/blob/main/figs/structure.png)

# The grad-cam results about CSFF
![image](https://github.com/zhangbaijin/Wakeup-Darkness/blob/main/figs/grad-cam.png)

# Compare results of LOL dataset
![image](https://github.com/zhangbaijin/Wakeup-Darkness/blob/main/figs/table.jpg)



## Model
`weights/`
`model.py` 
`fuse_block.py`

## Dataset processing
you should run the `sam.py` to produce the segment image like LOL/LIME/SCIE

## Train and Test

`python train.py --arc WithoutCalNet --batch_size 10`

`python test.py`

`python test_clip.py`

