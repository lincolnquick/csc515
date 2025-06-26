"""
CSC 515 - Module 2 Critical Thinking
Channel-split, merge, and swap on a color image (kitten)

Steps
1)  Import the colored image and split B, G, R channels - three nxn matrices
2)  Merge channels back in original order (BGR - RGB for display)
3)  Swap red - green to create a GRB composite and display result
Author: Lincoln Quick | 2025-06-19
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image and extract single-channel matrices
IMG_PATH = "kitten.jpg"        
img_bgr = cv2.imread(IMG_PATH)  # OpenCV loads BGR by default

if img_bgr is None:
    raise FileNotFoundError("Could not load image. Check the path.")

B, G, R = cv2.split(img_bgr)    # three n×n matrices
print(f"Channel matrix shape: {R.shape}")   # proves n×n

# Merge back to the original color image
merged_bgr = cv2.merge([B, G, R])             # still BGR
merged_rgb = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)

# Swap red and green, blue stays. Order: B, R, G instead of B, G, R
print("R-G swap: reddish features (nose, ears, wood in background) turn green, greenery shifts red (none present), edges stay intact, proving color channels can be permuted without altering geometry.")
grb_bgr = cv2.merge([B, R, G])
grb_rgb = cv2.cvtColor(grb_bgr, cv2.COLOR_BGR2RGB)

# Display results
#      Show in a 2 x 4 grid
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Turn off all axes tick marks
for ax in axes.flatten():
    ax.axis('off')

# Row 0: Original image
axes[0, 0].imshow(merged_rgb); axes[0, 0].set_title("Original")

# Row 0: Single-channel views
for c in range(3):
    split_img = np.zeros_like(img_bgr)
    split_img[:, :, c] = img_bgr[:, :, c]
    axes[0, c + 1].imshow(cv2.cvtColor(split_img, cv2.COLOR_BGR2RGB))
    axes[0, c + 1].set_title(["Blue", "Green", "Red"][c])

# Row 1: Merged images
axes[1, 0].imshow(merged_rgb); axes[1, 0].set_title("Merged RGB"); axes[1, 0].axis('off')
axes[1, 3].imshow(grb_rgb);    axes[1, 3].set_title("Swapped Red-Green"); axes[1, 3].axis('off')

# Save figure as image
OUT_FIG = "kitten_channel_grid.png"  
fig.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {os.path.abspath(OUT_FIG)}")

# Display image
plt.tight_layout();  plt.show(block=False)

# Close all windows after button press
print("Press any key in the figure window to close...")
plt.waitforbuttonpress()
plt.close('all')
cv2.destroyAllWindows()