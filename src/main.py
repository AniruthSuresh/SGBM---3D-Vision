import cv2
import numpy as np
import open3d as o3d
from sgbm import *
import matplotlib.pyplot as plt
# from sgm import *
import time as t

def get_the_intrinsics_matrix(scene_no  = 000000):
    """
    This function gets the intrinsics matrix of the current data from the file 
    /home/aniruth/Desktop/Courses/MR/Mid-Submission/SGBM/SGBM_Codes/data/data_stereo_flow/training/calib

    """
    calib_dir = "../data/data_stereo_flow/training/calib/"

    calib_file = f"{calib_dir}/{scene_no:06d}.txt"

    with open(calib_file, 'r') as file:
        lines = file.readlines()

    p2_line = next(line for line in lines if line.startswith("P2:"))
    
    p2_values = np.array([float(x) for x in p2_line.split()[1:]]).reshape(3, 4)
    
    K = p2_values[:, :3]

    print(K)
    return K




def load_disparity_map(left_name, right_name):
    """
    This function loads the stereo pair (left and right) images, computes the disparity map.
    """
    imgL = cv2.imread(left_name)
    imgR = cv2.imread(right_name)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=4)
    disparity = stereo.compute(grayL, grayR)

    return disparity, imgL

def process_stereo_images_opencv(left_name, right_name, evaluation=False, type="kitti"):
    """
    Call this function to process the stereo images and return the disparity map.
    """
    disparity, imgL = load_disparity_map(left_name, right_name)
    

    disp8 = np.uint8(disparity / np.max(disparity) * 255)
    if evaluation:
        pass

    maxdis = 220
    mindis = 35

    h , w , _ = np.array(cv2.imread(right_name)).shape

    # Ignore untrusted depth
    for i in range(h):
        for j in range(w):
            if (disp8[i][j] < mindis or disp8[i][j] > maxdis): disp8[i][j] = 0



    return disp8 

def disparity_to_pointcloud(disparity, img,scene_no ,  scale=1000):
    """
    Convert a disparity map to a 3D point cloud.

    Parameters:
        disparity (np.ndarray): Disparity map (grayscale or 8-bit unsigned).
        img (np.ndarray): RGB image corresponding to the disparity map.
        scale (float): Scaling factor for depth.
    """

    K = get_the_intrinsics_matrix(scene_no= scene_no)

    disparity = disparity.astype(np.float32)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]


    focal_length = fx # both fx and fy are same and = f here (square pixel)

    baseline = 379.8145 # from the text file 

    disparity = disparity.astype(np.float32)

    # depth = (focal_length * baseline/1000)  / (disparity + 1e-6)

    depth = 255 - disparity


    h, w = disparity.shape
    pc_points = []
    pc_colors = []


    for v in range(h):
        for u in range(w):

            if disparity[v, u] > 0: 

                z = depth[v, u]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                # print((x,y,z))
                pc_points.append([x, y, z])
                pc_colors.append(img[v, u] / 255.0)  # Normalize to [0,1]

    pc_points = np.array(pc_points, dtype=np.float32)
    pc_colors = np.array(pc_colors, dtype=np.float32)

    return pc_points, pc_colors




# def display_disparity_maps(disp_sgbm,disp_sgm_cv, disp_gt):
#     """
#     Plots the computed disp map and compares it with GT
#     """

#     plt.figure(figsize=(10, 5))

#     disparity_normalized = cv2.normalize(disp_gt, None, 0, 255, cv2.NORM_MINMAX)

#     disparity_normalized = np.uint8(disparity_normalized)

#     plt.subplot(1, 2, 1)
#     plt.imshow(disparity_normalized, cmap='gray')
#     plt.title('Original Disparity Map')
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.imshow(disp_sgbm, cmap='gray')
#     plt.title('SGBM Disparity Map')
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()


def display_disparity_maps(disp_sgbm, disp_sgm_cv, disp_gt):
    """
    Plots the computed disparity maps and compares them with ground truth (GT).
    """
    plt.figure(figsize=(15, 5))  # Increase figure width for 3 subplots

    # Normalize GT disparity map for visualization
    disparity_normalized = cv2.normalize(disp_gt, None, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # Plot GT disparity map
    plt.subplot(1, 3, 1)
    plt.imshow(disparity_normalized, cmap='gray')
    plt.title('Original Disparity Map (GT)')
    plt.axis('off')

    # Plot SGBM disparity map
    plt.subplot(1, 3, 2)
    plt.imshow(disp_sgbm, cmap='gray')
    plt.title('SGBM Disparity Map')
    plt.axis('off')

    # Plot SGM (CV) disparity map
    plt.subplot(1, 3, 3)
    plt.imshow(disp_sgm_cv, cmap='gray')
    plt.title('SGBM CV Disparity Map')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def calculate_mse(image1, image2):
    """
    Calculate the Mean Squared Error (MSE) between two images.

    Parameters:
        image1 (numpy.ndarray): First image (grayscale or color).
        image2 (numpy.ndarray): Second image (grayscale or color).

    Returns:
        float: The MSE value.
    """
    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Compute the Mean Squared Error
    mse = np.mean((image1 - image2) ** 2)
    return mse



scene_no = 5
count_of_scene =  0 
left_name = f"../data/data_stereo_flow/training/colored_0/{scene_no:06d}_1{count_of_scene}.png"
right_name=f"../data/data_stereo_flow/training/colored_1/{scene_no:06d}_1{count_of_scene}.png"
# left_name = f"../data/cones/im2.png"
# right_name= f"../data/cones/im6.png"

dawn = t.time()

disparity_left_cv = process_stereo_images_opencv(left_name, right_name,  evaluation=False, type="kitti") 
disparity_left = process_stereo_images(left_name, right_name, evaluation=False, type="kitti")

dusk = t.time()
disparity_left_gt = np.asarray(cv2.imread(f"../data/data_stereo_flow/training/disp_noc/{scene_no:06d}_1{count_of_scene}.png", cv2.IMREAD_GRAYSCALE))
# disparity_left_gt = np.asarray(cv2.imread(f"../data/cones/disp2.png", cv2.IMREAD_GRAYSCALE))


display_disparity_maps(disparity_left, disparity_left_cv,disparity_left_gt)


# disparity_normalized = cv2.normalize(disparity_left, None, 0, 255, cv2.NORM_MINMAX)

# disparity_normalized = np.uint8(disparity_normalized)


# disparity_normalized_cv = cv2.normalize(disparity_left_cv, None, 0, 255, cv2.NORM_MINMAX)
# disparity_normalized_cv = np.uint8(disparity_normalized_cv)

# disparity_normalized_gt = cv2.normalize(disparity_left_gt, None, 0, 255, cv2.NORM_MINMAX)
# disparity_normalized_gt = np.uint8(disparity_normalized_gt)
accuracy = calculate_mse(disparity_left,disparity_left_gt)
# accuracy_cv = calculate_mse(disparity_left_cv,disparity_left_gt)

# pc_points_1, pc_colors_1 = disparity_to_pointcloud(disparity_left_gt, cv2.imread(left_name) , scene_no = scene_no)
pc_points, pc_colors = disparity_to_pointcloud(disparity_left, cv2.imread(left_name) , scene_no = scene_no)
pc_points2, pc_colors2 = disparity_to_pointcloud(disparity_left_cv, cv2.imread(left_name) , scene_no = scene_no)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.colors = o3d.utility.Vector3dVector(pc_colors)

pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(pc_points2)
pcd_2.colors = o3d.utility.Vector3dVector(pc_colors2)

# pcd_1 = o3d.geometry.PointCloud()
# pcd_1.points = o3d.utility.Vector3dVector(pc_points_1)
# pcd_1.colors = o3d.utility.Vector3dVector(pc_colors_1)

# o3d.visualization.draw_geometries([pcd_1],
#                                   zoom=0.0412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.0412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])



o3d.visualization.draw_geometries([pcd_2],
                                  zoom=0.0412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])




print(f"MSE and time taken for SGBM: {accuracy} , {dusk - dawn} seconds")
# print(f"MSE for SGBM_CV {accuracy_cv}")



