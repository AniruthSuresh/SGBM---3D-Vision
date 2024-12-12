# SGBM - 3D reconstruction 

This repository showcases the implementation of the **Semi-Global Block Matching (SGBM)** algorithm, a widely used method in stereo vision for computing disparity maps. The project also extends to reconstructing 3D scenes from the computed disparity maps . 

## 1. About SGBM

The **Semi-Global Block Matching (SGBM)** algorithm is an efficient approach to compute dense disparity maps for stereo image pairs. It combines the accuracy of global methods with the efficiency of local methods, striking a balance suitable for real-world applications. 

### Key Steps:
1. **Cost Computation:** Matching cost is computed for each pixel in the stereo pair.
2. **Cost Aggregation:** SGBM uses paths in multiple directions to minimize the disparity cost semi-globally.
3. **Disparity Computation:** The disparity for each pixel is chosen by minimizing the aggregated cost.
4. **Post-Processing:** Filters such as median and bilateral can be applied to refine the disparity map.

### 3D Reconstruction
Using the disparity map, depth information is extracted by applying the following formula:

$$Z = \frac{f \cdot B}{d}$$


where:
- \( Z \): Depth of the point in the scene.
- \( f \): Focal length of the camera.
- \( B \): Baseline distance between the stereo cameras.
- \( d \): Disparity value.

The resulting depth map is used to reconstruct the 3D scene by mapping points into the camera’s 3D coordinate space.

---
## 2. Datasets

### 1. SGBM
- For the SGBM project, we used the KITTI stereo dataset.  
  Download the dataset from the official KITTI website:  
  [KITTI Stereo Dataset](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)

---

## 3. Installation

Set up the environment using the steps below:

```bash
conda create --name myenv
conda activate myenv
conda install numpy opencv matplotlib
```
---
## 4. Usage

- The SGBM project uses only basic Python libraries and does not require a complex setup.
  
- **Run the code**:
    ```bash
    cd src
    python check.py # This is the main file
    ```
- Ensure the KITTI dataset is placed in the specified directory before execution.


