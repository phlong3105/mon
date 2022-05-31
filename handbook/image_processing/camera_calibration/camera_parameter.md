<div align="center">
<br><br>
<div>
	<a href="https://github.com/phlong3105/one/blob/master/handbook/README.md"><img src="../../data/badge/handbook_home.svg"></a>
    <img src="../../data/badge/handbook_image_processing.svg">
</div>

What are Intrinsic and Extrinsic Camera Parameters in Computer Vision?
=============================

<div align="center">
    <a href="https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec#:~:text=The%20extrinsic%20matrix%20is%20a,to%20the%20pixel%20coordinate%20system."><img src="../../data/badge/paper_website.svg"></a>
</div>
</div>


## Highlight
- Cameras are the sensors used to capture images. 
  - They take the points in the world and project them onto a 2D plane which we see as images.


- The complete transformation is usually divided into two parts: **Extrinsic** and **Intrinsic**. 
  - The extrinsic parameters depend on its location and orientation and have nothing to do with its internal parameters such as focal length, the field of view, etc. 
  - The intrinsic parameters of a camera depend on how it captures the images. Parameters such as focal length, aperture, field-of-view, resolution, etc. govern the intrinsic matrix of a camera model.

<div align="center">
    <img width="700" src="data/transformation.png"><br/>
</div>


## Method
- Extrinsic and extrinsic parameters are transformation matrices that convert points from one coordinate system to the other. 
- In order to understand these transformations, we first need to understand **four different coordinate systems**:
  - World coordinate system (3D)
  - Camera coordinate system (3D)
  - Image coordinate system (2D)
  - Pixel coordinate system (2D)


<details open>
<summary><b style="font-size:16px">1. World coordinate system (3D)</b></summary>

<div align="center">
    <img width="600" src="data/world_coordinate_system.png"><br/>
</div>

- **[Xw, Yw, Zw]**: It is a 3D basic cartesian coordinate system with arbitrary origin. 
- For example a specific corner of the room. A point in this coordinate system can be denoted as Pw = (Xw, Yw, Zw).

</details>

<details open>
<summary><b style="font-size:16px">2. Object/Camera coordinate system (3D)</b></summary>

<div align="center">
    <img width="600" src="data/camera_coordinate_system.png"><br/>
</div>

- **[Xc, Yc, Zc]**: It's the coordinate system that measures relative to the object/camera’s origin/orientation. 
- The z-axis of the camera coordinate system usually faces outward or inward to the camera lens (camera principal axis) as shown in the image above (z-axis facing inward to the camera lens). 

<div align="center">
    <img width="600" src="data/camera_extrinsic_matrix.png"><br/>
</div>

- One can go from the world coordinate system to object coordinate system (and vice-versa) by Rotation and Translation operations.
  - The 4x4 transformation matrix that converts points from the world coordinate system to the camera coordinate system is known as the camera extrinsic matrix.
  - The camera extrinsic matrix changes if the physical location/orientation of the camera is changed (for example camera on a moving car).

</details>

<details open>
<summary><b style="font-size:16px">3. Image coordinate system (2D) [Pinhole Model]</b></summary>

<div align="center">
    <img width="600" src="data/image_coordinate_system.png"><br/>
</div>

- **[Xi, Yi]**: A 2D coordinate system that has the 3D points in the camera coordinate system projected onto a 2D plane (usually normal to the z-axis of the camera coordinate system — shown as a yellow plane in the figures below) of a camera with a Pinhole Model.
  - The rays pass the center of the camera opening and are projected on the 2D plane on the other end. 
  - The 2D plane is what is captured as images by the camera. 
  - It is a lossy transformation, which means projecting the points from the camera coordinate system to the 2D plane can not be reversed (the depth information is lost — Hence by looking at an image captured by a camera, we can’t tell the actual depth of the points). 
  - The X and Y coordinates of the points are projected onto the 2D plane. 
  - The 2D plane is at f (focal-length) distance away from the camera. 
  - The projection Xi, Yi can be found by the law of similar triangles (the ray entering and leaving the camera center has the same angle with the x and y-axis, alpha and beta respectively).

<div align="center">
    <img width="600" src="data/pinhole_camera_01.png"><br/>
    <img width="600" src="data/pinhole_camera_02.png"><br/>
    <img width="600" src="data/pinhole_camera_03.png"><br/>
    
</div>

- Hence in the matrix form, we have the following transformation matrix from the camera coordinate system to the image coordinate system.
- This transformation (from camera to image coordinate system) is the first part of the camera intrinsic matrix

<div align="center">
  <img width="600" src="data/pinhole_camera_04.png"><br/>
</div>

</details>

<details open>
<summary><b style="font-size:16px">4. Pixel coordinate system (2D)</b></summary>

<div align="center">
    <img width="600" src="data/pixel_coordinate_system_01.png"><br/>
</div>

- **[u, v]**: This represents the integer values by discretizing the points in the image coordinate system. 
- Pixel coordinates of an image are discrete values within a range that can be achieved by dividing the image coordinates by pixel width and height (parameters of the camera — units: meter/pixel).


- The pixel coordinates system has the origin at the left-top corner, hence a translation operator (c_x, c_y) is also required alongside the discretization.

<div align="center">
    <img width="600" src="data/pixel_coordinate_system_02.png"><br/>
</div>

- The complete transformation from the image coordinate system to pixel coordinate system can be shown in the matrix form as below.

<div align="center">
    <img width="600" src="data/pixel_coordinate_system_03.png"><br/>
</div>

- Sometimes, the 2D image plane is not a rectangle but rather is skewed i.e. the angle between the X and Y axis is not 90 degrees. 
- In this case, another transformation needs to be carried out to go from the rectangular plane to the skewed plane (before carrying out the transformation from image to pixel coordinate system). 
- If the angle between the x and y-axis is theta, then the transformation that converts points from the ideal rectangular plane to the skewed plane can be found as below

<div align="center">
    <img width="600" src="data/pixel_coordinate_system_04.png"><br/>
</div>

- These two transformation matrices i.e. **transformation from rectangular image coordinate system to skewed image coordinate system and skewed image coordinate system to pixel coordinate system** forms the second part of the **camera intrinsic matrix**. 
- Combining the three transformation matrices yields the camera extrinsic matrix as shown below

<div align="center">
    <img width="600" src="data/camera_intrinsic_matrix_01.png"><br/>
    <img width="600" src="data/camera_intrinsic_matrix_02.png"><br/>
    <img width="600" src="data/camera_intrinsic_matrix_03.png"><br/>
</div>


</details>

### Summary
- The extrinsic matrix is a transformation matrix from the world coordinate system to the camera coordinate system, while the intrinsic matrix is a transformation matrix that converts points from the camera coordinate system to the pixel coordinate system.
  - **World-to-Camera**: 3D-3D projection. Rotation, Scaling, Translation 
  - **Camera-to-Image**: 3D-2D projection. Loss of information. Depends on the camera model and its parameters (pinhole, f-theta, etc)
  - **Image-to-Pixel**: 2D-2D projection. Continuous to discrete. Quantization and origin shift.
