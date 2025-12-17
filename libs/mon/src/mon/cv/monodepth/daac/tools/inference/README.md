# Tools Usage Guide

**English** | [中文](README_zh.md)

This directory contains tool scripts for the DepthAnything-AC project.

## Inference Script (infer.py)

The inference script supports processing of both images and videos, with flexible batch processing capabilities.

### Basic Usage

#### Single Image Inference
```bash
# Basic usage
python tools/infer.py --input image.jpg --output depth.png

# Specify model and encoder
python tools/infer.py --input image.jpg --output depth.png --model checkpoints/depth_anything_AC_vits.pth --encoder vits

# Use different colormap
python tools/infer.py --input image.jpg --output depth.png --colormap inferno
```

#### Single Video Inference
```bash
# Basic video processing
python tools/infer.py --input video.mp4 --output depth_video.mp4

# Specify output FPS
python tools/infer.py --input video.mp4 --output depth_video.mp4 --fps 30

# Use different colormap for video
python tools/infer.py --input video.mp4 --output depth_video.mp4 --colormap spectral
```

#### Batch Processing
```bash
# Process images only in a directory
python tools/infer.py --input images/ --output output/ --mode images

# Process videos only in a directory
python tools/infer.py --input videos/ --output output/ --mode videos

# Process both images and videos (default mode)
python tools/infer.py --input media/ --output output/ --mode mixed

# Recursive processing of subdirectories
python tools/infer.py --input dataset/ --output results/ --recursive
```

### Parameters

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | str | - | Input image/video path or directory (required) |
| `--output` | `-o` | str | - | Output path (file or directory) (required) |
| `--model` | `-m` | str | `checkpoints/depth_anything_v2_vits.pth` | Model weight path |
| `--encoder` | - | str | `vits` | Encoder type (`vits`, `vitb`, `vitl`) |
| `--colormap` | - | str | `spectral` | Colormap (`inferno`, `spectral`, `gray`) |
| `--depth_cap` | - | float | `80` | Maximum depth value for capping |
| `--fps` | - | float | `None` | Output video FPS (defaults to input FPS) |
| `--recursive` | `-r` | flag | `False` | Search recursively in subdirectories |
| `--mode` | - | str | `mixed` | Processing mode (`images`, `videos`, `mixed`) |

### Supported File Formats

**Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

**Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.m4v`

### Usage Examples

#### Example 1: Process Single Image
```bash
python tools/infer.py -i test_image.jpg -o result.png
```
Output:
- `result.png` - Colored depth map
- `result_raw.npy` - Raw depth data

#### Example 2: Process Single Video
```bash
python tools/infer.py -i test_video.mp4 -o depth_video.mp4 --fps 30
```
Output:
- `depth_video.mp4` - Video with depth visualization

#### Example 3: Batch Process Images Only
```bash
python tools/infer.py -i photos/ -o depth_results/ --mode images --recursive
```
Recursively processes all images in the `photos/` directory and subdirectories, generating corresponding depth maps in the `depth_results/` directory.

#### Example 4: Batch Process Videos Only
```bash
python tools/infer.py -i videos/ -o depth_videos/ --mode videos --fps 25
```
Processes all videos in the `videos/` directory, generating depth videos with 25 FPS output.

#### Example 5: Mixed Batch Processing
```bash
python tools/infer.py -i media/ -o output/ --mode mixed --colormap inferno
```
Processes both images and videos in the `media/` directory, using inferno colormap for visualization.

### Output Structure

When processing directories, the output maintains the same folder structure as the input:
- Images: `input_name_depth.png` + `input_name_depth_raw.npy`
- Videos: `input_name_depth.mp4`

## Training Scripts

### train.sh
Script for launching distributed training.

```bash
bash tools/train.sh [NUM_GPUS] [PORT]
```

### val.sh  
Script for launching distributed evaluation.

```bash
bash tools/val.sh [NUM_GPUS] [PORT] [DATASET]
```

---

For more detailed information, please refer to the main project [README](../README.md) documentation. 
