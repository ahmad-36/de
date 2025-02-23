# ** 3D Reconstruction Using Stereo Disparity**
You will learn about 3D reconstruction using stereo vision. You’ll perform stereo rectification, stereo matching, disparity calculation, and 3D point cloud generation. Each section contains theoretical background and code outlines with function placeholders, along with hints on OpenCV functions to use.

I am uploading three datasets:

1. Bike
2. Umbrella
3. Cycle

These datasets are taken from [Middlebury](https://vision.middlebury.edu/stereo/data/scenes2014/) and contain the left and right images along with the calibration information.

---

## **Calibration File Format**

Here is a sample `calib.txt` file for one of the full-size training image pairs:

```
cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
doffs=131.111
baseline=193.001
width=2964
height=1988
ndisp=280
isint=0
vmin=31
vmax=257
dyavg=0.918
dymax=1.516
```

### **Explanation**
- **cam0, cam1**: Camera matrices for the rectified views, in the form `[f 0 cx; 0 f cy; 0 0 1]` where:
  - `f`: focal length in pixels
  - `cx, cy`: principal point coordinates (note that `cx` differs between view 0 and 1)
- **doffs**: x-difference of principal points, `doffs = cx1 - cx0`
- **baseline**: Camera baseline in mm
- **width, height**: Image dimensions
- **ndisp**: Upper bound on disparity levels
- **isint**: Whether ground-truth disparities are integer-valued
- **vmin, vmax**: Minimum and maximum disparities for visualization
- **dyavg, dymax**: Indicators of calibration error

---

## **Objective**

### **Section 1: Stereo Rectification**

#### **Theory**
Stereo rectification ensures that corresponding points in the two images lie on the same scanline, simplifying disparity calculation. 

#### **Code Outline and Hints**
```python
import cv2

def stereo_rectify(K0, K1, R, T, image_size):
    """
    Perform stereo rectification.
    :param K0: Intrinsic matrix of left camera
    :param K1: Intrinsic matrix of right camera
    :param R: Rotation matrix between the cameras
    :param T: Translation vector between the cameras
    :param image_size: Tuple (width, height)
    :return: Rectification maps
    """
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K0, K1, None, None, image_size, R, T)
    return R1, R2, P1, P2, Q
```

- **Functions to Explore**: `cv2.stereoRectify`

---

### **Section 2: Stereo Matching and Disparity Map Calculation**

#### **Theory**
Stereo matching computes disparity by finding corresponding pixels between left and right images. Depth $Z$ is derived as:
$$
Z = \frac{f \cdot B}{d}
$$
where $f$ is the focal length, $B$ is the baseline, and $d$ is the disparity.

#### **Code Outline and Hints**
```python
import numpy as np

def compute_disparity_map(imgL, imgR):
    """
    Compute disparity map from stereo images.
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*16,
        blockSize=15,
        P1=8*3*15**2,
        P2=32*3*15**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity_map = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disparity_map
```

- **Functions to Explore**: `cv2.StereoBM_create`, `cv2.StereoSGBM_create`

---

### **Section 3: 3D Reconstruction**

#### **Theory**
The disparity map enables 3D reconstruction by computing $(X, Y, Z)$ coordinates for each pixel:
$$
X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = \frac{f \cdot B}{d}
$$
where $(u, v)$ is the pixel position, and $(c_x, c_y)$ are principal points.

#### **Code Outline and Hints**
```python
import cv2
import numpy as np

def reconstruct_3D(disparity_map, Q):
    """
    Reconstruct 3D points from disparity map.
    """
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3D
```

- **Functions to Explore**: `cv2.reprojectImageTo3D`

---

### **Section 4: Visualization**

#### **Theory**
To verify reconstruction, visualize the 3D points using Open3D.

#### **Code Outline and Hints**
```python
import open3d as o3d

def visualize_point_cloud(points_3D, colors):
    """
    Visualize the 3D point cloud.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3D.reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255.0)
    o3d.visualization.draw_geometries([point_cloud])
```

- **Functions to Explore**: `o3d.geometry.PointCloud`, `o3d.visualization.draw_geometries`

---

## **References**
- [OpenCV Depth Map Tutorial](https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_depthmap.html)
