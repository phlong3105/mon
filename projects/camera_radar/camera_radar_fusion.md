# Camera-Radar Fusion for Vehicle Speed Estimation

---

## Introduction

This manual guides you through a project to estimate vehicle speeds using a camera and radar in a static surveillance setup. You will use YOLO to detect vehicles in camera images and link them to radar data for speed values.

The setup assumes a fixed camera and radar overlooking a road. Radar gives direct speed measurements, while the camera provides visual detection. Assume normal traffic conditions (e.g., no congestion or heavily occluded vehicles).

**Project Goals:** 

* Detect & track vehicles with YOLO + SORT. 
* Associate detections with radar speeds. 
* Output bounding boxes with speed values.

---

## Requirements

### Hardware

* Camera (e.g., USB webcam or IP camera)
* Radar: mmWave radar sensor that outputs point clouds with velocity. 
* Computer: With Python support, webcam/radar ports. 
* Calibration targets: Checkerboard (for camera) with corner reflectors (for radar).

### Software

* Python 3.8+
* OpenCV: For calibration.
* Ultralytics: For YOLO.
* SORT: For tracking. URL: https://github.com/abewley/sort
* Radar SDK: From the manufacturer.
* Numpy, SciPy, scikit-learn.

---

## Step-by-Step Instructions

### Step 1: Set Up Environment

- Install Python and libraries:
```bash
pip install opencv-python ultralytics numpy scipy scikit-learn
``` 

- Connect hardware: Plug in camera and radar. Test camera with `OpenCV`. Test radar with its SDK to get point clouds.

### Step 2: Calibrate Camera and Radar

**Goal:** Calibration aligns camera and radar sensors for data fusion.

Camera intrinsic calibration:

- Print a checkerboard (e.g., 9x6 squares).
- Capture 10-20 images from different angles.
- Use OpenCV: `ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)`
- Save matrix and distortion coefficients.

Extrinsic calibration (radar-camera pose):

- Checkerboard approach:
  - Place corner reflectors in view of both sensors.
  - Record positions in radar (range, azimuth) and camera (pixels).
  - Use OpenCV `cv2.solvePnP` to compute rotation and translation matrix.
  - Save the extrinsic matrix.

- `OpenCalib` approach:
  - `OpenCalib` is a toolbox for sensor calibration in autonomous driving. It supports camera-radar extrinsic calibration.
  - URL: https://github.com/PJLab-ADG/SensorsCalibration/tree/master/radar2camera

Test: Project a known radar point to camera image; it should match visually.

### Step 3: Detect and Track Vehicles

- Load pre-trained YOLOv8 model: `model = YOLO('yolov8n.pt')`.
- Process camera frame: `results = model(frame)`.
- Extract bounding boxes: For each detection, get `[x, y, width, height]` if class is `'car'` or `'truck'`. Filter confidence scores` > 0.5`.
- Track detections: Use SORT tracker. 
  ```python
  from sort import *
   
  # Initialize tracker
  mot_tracker = Sort()
    
  # Update tracker with detections
  track_bbs_ids = mot_tracker.update(detections_array)
  ```
    , where detections_array is NumPy array of `[x1, y1, x2, y2, score]`.
    
Output: Tracked bounding boxes with IDs per frame.

### Step 4: Collect and Process Radar Data

- Use radar SDK to get point clouds: Each point has range, azimuth, radial velocity.
- Cluster points: Use scikit-learn DBSCAN:
  ```python
  from sklearn.cluster import DBSCAN
  
  clusters = DBSCAN(eps=0.5, min_samples=3).fit(points)
  ```

### Step 5: Project Radar Points to Image Plane

- Convert radar points to 3D: Use range and azimuth (assume elevation=0 for flat road).
- Apply extrinsic `matrix: projected = np.dot(extrinsic, radar_3d)`
- Apply intrinsic matrix: `pixel = np.dot(intrinsic, projected) / projected[2]`

Output: Radar points in pixel coordinates.

### Step 6: Associate Tracked Bounding Boxes with Radar Points

- For each tracked box, find radar points inside it: Check if `pixel_x > box_x and < box_x + width`, etc.
- Use distance threshold: If points are near (e.g., < 10 pixels), associate.
- Average velocities: Assign mean radial velocity to the tracked box. Use track ID for persistence.

### Step 7: Estimate Speed

- Radial velocity is line-of-sight speed.
- Compute magnitude: `speed = radial_velocity / cos(angle)`, where `angle` is between radar line and vehicle direction.

Output: Tracked bounding box with ID and speed value (e.g., km/h).

### Step 8: Output Results

- Draw on frame: Use OpenCV `cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)`
- Add text: `cv2.putText(frame, f'ID: {track_id} Speed: {speed} km/h', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)`
- Display or save.

---

## Common Issues

- Misalignment: Recalibrate if projections are off.
- Lost tracks: Adjust SORT parameters (e.g., `max_age`).
- Radar noise: Increase clustering `min_samples`.
