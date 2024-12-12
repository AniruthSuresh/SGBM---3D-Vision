# SGBM - Mid-Submission Report

This repository showcases the implementation of the **Semi-Global Block Matching (SGBM)** algorithm, a widely used method in stereo vision for computing disparity maps. The project also extends to reconstructing 3D scenes from the computed disparity maps, making it a crucial step in stereo vision applications.

---

## 1. About SGBM

The **Semi-Global Block Matching (SGBM)** algorithm is an efficient approach to compute dense disparity maps for stereo image pairs. It combines the accuracy of global methods with the efficiency of local methods, striking a balance suitable for real-world applications. 

### Key Steps:
1. **Cost Computation:** Matching cost is computed for each pixel in the stereo pair.
2. **Cost Aggregation:** SGBM uses paths in multiple directions to minimize the disparity cost semi-globally.
3. **Disparity Computation:** The disparity for each pixel is chosen by minimizing the aggregated cost.
4. **Post-Processing:** Filters such as median and bilateral can be applied to refine the disparity map.

### 3D Reconstruction
Using the disparity map, depth information is extracted by applying the following formula:
\[
Z = \frac{f \cdot B}{d}
\]

where:
- \( Z \): Depth of the point in the scene.
- \( f \): Focal length of the camera.
- \( B \): Baseline distance between the stereo cameras.
- \( d \): Disparity value.

The resulting depth map is used to reconstruct the 3D scene by mapping points into the camera’s 3D coordinate space.

---

## 2. Installation

Set up the environment using the steps below:

```bash
conda create --name myenv
conda activate myenv
conda install numpy opencv matplotlib
```

