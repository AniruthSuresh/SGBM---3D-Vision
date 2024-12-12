import matplotlib.pyplot as plt
import cv2

# Load your screenshots
# Replace these file paths with the paths to your screenshot images
sgbm_image = cv2.imread('/home/aadi_iiith/Desktop/MR/SGBM (copy)/SGBM_Codes/results/kitti_sgbm.png')  # SGBM reconstruction screenshot
sgbm_cv_image = cv2.imread('/home/aadi_iiith/Desktop/MR/SGBM (copy)/SGBM_Codes/results/kitti_cv.png')  # SGBM CV reconstruction screenshot

# Convert BGR to RGB for proper display
sgbm_image = cv2.cvtColor(sgbm_image, cv2.COLOR_BGR2RGB)
sgbm_cv_image = cv2.cvtColor(sgbm_cv_image, cv2.COLOR_BGR2RGB)

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# SGBM Reconstruction
axes[0].imshow(sgbm_image)
axes[0].set_title("SGBM Reconstruction")
axes[0].axis('off')  # Hide axes for better visualization

# SGBM CV Reconstruction
axes[1].imshow(sgbm_cv_image)
axes[1].set_title("SGBM CV Reconstruction")
axes[1].axis('off')  # Hide axes for better visualization

# Show the plot
plt.tight_layout()
plt.show()
