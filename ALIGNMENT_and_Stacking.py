import cv2
import numpy as np
import os

# 1. Setup paths and load images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(BASE_DIR, 'keyboard1.jpeg')
img2_path = os.path.join(BASE_DIR, 'keyboard2.jpeg')

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not locate keyboard1.tif or keyboard2.tif in your directory.")

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 2. Extract keypoints and compute descriptors (using SIFT)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3. Match features using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test to filter out false matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

if len(good_matches) < 4:
    raise ValueError("Not enough matching features found to stitch images.")

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 4. Find Homography Matrix
H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

# 5. Dynamically calculate the dimensions of the stitched canvas
# Get the corners of Image 2 to see where they land after transformation
corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)

# Combine corners of Image 1 and warped Image 2 to find overall bounds
corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
all_corners = np.concatenate((corners_img1, warped_corners_img2), axis=0)

[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Create a translation matrix to shift everything into positive coordinates
translation_dist = [-x_min, -y_min]
H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

# 6. Warp and stitch onto the mega canvas
output_width = x_max - x_min
output_height = y_max - y_min

# Warp Image 2 onto the canvas
stitched_img = cv2.warpPerspective(img2, H_translation.dot(H), (output_width, output_height))

# Generate individual masks on the final canvas size to locate the overlap region
mask1 = np.zeros((output_height, output_width), dtype=np.uint8)
# Place Image 1 on the canvas
stitched_img[translation_dist[1]:h1+translation_dist[1], translation_dist[0]:w1+translation_dist[0]] = img1
mask1[translation_dist[1]:h1+translation_dist[1], translation_dist[0]:w1+translation_dist[0]] = 255

# Re-warp image 2 just as a mask to find its exact final footprint
mask2_raw = np.ones((h2, w2), dtype=np.uint8) * 255
mask2 = cv2.warpPerspective(mask2_raw, H_translation.dot(H), (output_width, output_height))

# 7. Extract overlap boundaries and outline it
overlap_mask = cv2.bitwise_and(mask1, mask2)
contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw a red line (BGR: 0, 0, 255) around the shared overlap zone
cv2.drawContours(stitched_img, contours, -1, (0, 0, 255), 3)

# 8. Save stitched result
output_path = os.path.join(BASE_DIR, 'stitched_with_overlap.png')
cv2.imwrite(output_path, stitched_img)

print(f"Stitching complete! Both images combined with overlap highlighted at: {output_path}")