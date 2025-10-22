<h1 align="center">[ICCV 2025] MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices</h1>

<div align="center">
  <hr>
  Hailong Yan<sup>1</sup>&nbsp;
  Ao Li<sup>1</sup>&nbsp;
  Xiangtao Zhang<sup>1</sup>&nbsp;
  Zhe Liu<sup>1</sup>&nbsp;
  Zenglin Shi<sup>2</sup>&nbsp;
  Ce Zhu<sup>1</sup>&nbsp;
  Le Zhang<sup>1,†</sup>&nbsp;
  <br>
  <sup>1</sup> UESTC&nbsp;&nbsp; <sup>2</sup> Hefei University of Technology<br>
  <sup>†</sup> Corresponding author.<br>

  <h4>
    <a href="https://www.arxiv.org/pdf/2507.01838">📄 arXiv Paper</a> &nbsp; 
  </h4>
</div>

<blockquote>
<b>Abstract:</b> <i>Recent advancements in deep neural networks have driven significant progress in image enhancement (IE). However, deploying deep learning models on resource-constrained platforms, such as mobile devices, remains challenging due to high computation and memory demands. To address these challenges and facilitate real-time IE on mobile, we introduce an extremely lightweight Convolutional Neural Network (CNN) framework with around 4K parameters. Our approach integrates reparameterization with an Incremental Weight Optimization strategy to ensure efficiency. Additionally, we enhance performance with a Feature Self-Transform module and a Hierarchical Dual-Path Attention mechanism, optimized with a Local Variance-Weighted loss. With this efficient framework, we are the first to achieve real-time IE inference at up to 1,100 frames per second (FPS) while delivering competitive image quality, achieving the best trade-off between speed and performance across multiple IE tasks.</i>
</blockquote>

<p align="center">
  <img width="1000" src="figs/framework.png">
</p>

---


### Preparation

1. Replace the dataset path in the config file.
2. If you want to train the model, set the type in config to "original" and need_slims to "false".
3. If you want to test the pretrain model, set the type in config to "re-parameterized", need_slims to "true", and load the re-parameterized pre-trained model. You can also run inference with TFLite model by executing "test_TFLite_RGB.py/test_TFLite_ISP.py".
4. You can use the TFLite model and import it into AI Benchmark (https://ai-benchmark.com/) to obtain the inference speed on mobile devices.
5. If you want to perform UIE task, replace the dataset path in config/lle.yaml with your underwater image dataset.

### Train

```bash
python main.py -task train -model_task lle/isp -device cuda
```

### Test

```bash
python main.py -task test -model_task lle/isp -device cuda
```

### Demo

```bash
python main.py -task demo -model_task lle/isp -device cuda
```

### Contact
If you have any questions, please contact me by e-mail (yanhailong@std.uestc.edu.cn; yhl00825@163.com).

### Citation

If you find the code helpful in your research or work, please cite the following paper:

```
@InProceedings{yan2025mobileie,
    author    = {Yan, Hailong and Li, Ao and Zhang, Xiangtao and Liu, Zhe and Shi, Zenglin and Zhu, Ce and Zhang, Le},
    title     = {MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    month     = {October},
    year      = {2025},
}
```
