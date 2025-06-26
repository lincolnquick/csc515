"""
CSC515 - Module 2 Discussion Demo
Geometric normalization + contrast enhancement for counterfeit detection
Lincoln Quick | 2025-06-19
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Load the image and inspect basic properties  (Module 2.3)
# ────────────────────────────────────────────────────────────────────────────────

IMG_PATH = 'banknotes.jpg'    
img_bgr = cv2.imread(IMG_PATH)

if img_bgr is None:
    raise FileNotFoundError(f'Could not load {IMG_PATH}')

h, w, c = img_bgr.shape
print(f'Original shape  : {img_bgr.shape}  (HxWxC)')
print(f'Pixel[50,100] BGR values : {img_bgr[50, 100]}')

# OpenCV uses BGR; convert to RGB for Matplotlib preview
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ────────────────────────────────────────────────────────────────────────────────
# 2.  Affine normalisation pipeline  (Module 2.2)
#     • Translation  – here we keep (tx, ty) = (0, 0) as per instructor note
#     • Rotation     – rotate –90 ° about center
#     • Scaling      – resize so width = 800 px, aspect preserved
# ────────────────────────────────────────────────────────────────────────────────

# Translation matrix
tx, ty = 0, 0
T = np.float32([[1, 0, tx],
                [0, 1, ty]])

# Rotation matrix about the image center
center = (w / 2, h / 2)
angle = -90       
scale = 1.0
R = cv2.getRotationMatrix2D(center, angle, scale)

# Apply translation -> rotation 
translated = cv2.warpAffine(img_bgr, T,   (w, h))
rotated    = cv2.warpAffine(translated, R, (w, h))

# Uniform scaling so final width = 800 px
target_w   = 800
scale_fac  = target_w / rotated.shape[1]
new_size   = (target_w, int(rotated.shape[0] * scale_fac))
scaled     = cv2.resize(rotated, new_size, interpolation=cv2.INTER_LINEAR)

# ────────────────────────────────────────────────────────────────────────────────
# 3.  Image‐enhancement pipeline
#     • Unsharp masking      – emphasise fine edges / micro-printing
#     • LAB + CLAHE on L-channel  – local contrast boost, thread/serial clarity
# ────────────────────────────────────────────────────────────────────────────────

# Unsharp mask: original + (original – GaussianBlur)
blur = cv2.GaussianBlur(scaled, (9, 9), sigmaX=3)
sharpened = cv2.addWeighted(scaled, 1.5, blur, -0.5, 0)

# Convert to LAB and apply CLAHE on the luminosity (L) channel
lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
L_eq  = clahe.apply(L)

lab_eq = cv2.merge([L_eq, A, B])
enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# ────────────────────────────────────────────────────────────────────────────────
# 4.  Display & save results
# ────────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(img_rgb);              ax[0].set_title('Original');   ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
ax[1].set_title('Normalized + Enhanced');       ax[1].axis('off')
plt.tight_layout();  plt.show()

OUT_PATH = 'banknotes_transformed.jpg'
cv2.imwrite(OUT_PATH, enhanced)
print(f'Enhanced image saved to → {os.path.abspath(OUT_PATH)}')