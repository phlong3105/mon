# Image Debanding using Inversion by Direct Iteration 
This repository implements training and tools to deband images and videos using generative models.

https://github.com/ksasso1028/indi-debanding/assets/11267645/4def7b00-4bc7-4ca0-9c13-faae985c3c63

# Examples
Full resolution results can be seen in the examples folder, network can handle variable size inputs without preprocessing.
![deband-collage](https://github.com/ksasso1028/indi-debanding/assets/11267645/f18d3d18-dba4-4b16-b8ca-c6b89ea45860)

# Introduction to Image Banding
Image banding is a visual artifact that appears when smooth color gradients are displayed with insufficient color depth, resulting in visible lines or "bands" instead of a seamless transition. This effect is particularly noticeable in areas with gradual color changes, such as skies or shadows. Typically affects cameras with lower bit depth (8 bit footage)

## The Problem
Banding can occur due to various reasons like image compression, low bit-depth formats, or improper processing techniques. It detracts from the visual quality of images and can be problematic in fields requiring high-precision visuals, such as photography, medical imaging, and digital art.

## Solution
This repository offers a solution for image debanding utilizing a technique known as ["Inversion by Direct Iteration"](https://arxiv.org/abs/2303.11435) (INDI) created to handle inverse problems. INDI restores the degraded image in a series of small steps, similar to conditional diffusion models. These models help smooth out gradients and reduce the appearance of bands while maintaining the original details and textures of the image. However instead of applying INDI to the raw signal, we apply it to the 2D FFT of the outputs to enforce restoration across the frequency space. This approach can restore images of variable sizes, with the same model. 

TODO:

- [ ] upload script to process videos with FFMPEG and avoid re encoding new video with debanded frames
- [x] upload model weights
- [ ] discuss dataset used + metrics
