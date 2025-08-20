import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
import nibabel as nib
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN
from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from collections import deque
from skimage.measure import label, regionprops


# === Load image slice ===
nifti_file = nib.load(r"C:\Users\arkaniva\Projects\Master-Thesis-BrachyTherapy\image2.nii")
image_data = nifti_file.get_fdata()
vol = np.zeros_like(image_data, dtype=image_data.dtype) # Ensure float32 for processing
for i in range(image_data.shape[2]):
    vol[:, :, i] = image_data[:, :, i].T
image_data = vol




import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from scipy.ndimage import gaussian_gradient_magnitude
import skfuzzy as fuzz
from skimage import filters
from skimage.measure import label
from skimage import filters
from skimage.filters import gaussian
from skimage.filters.rank import entropy
from skimage.util import img_as_float, img_as_ubyte
from skimage.morphology import disk
from scipy.ndimage import uniform_filter
from skimage.filters import sobel
import skfuzzy as fuzz
import numpy as np
from sklearn.decomposition import PCA



def segment_needle_slice(image, cluster_n=4):
    image = img_as_float(image)
    print(f"Segmenting slice with shape: {[np.min(image), np.max(image)]}")
    h, w = image.shape

    

    # Mask to exclude low-intensity background
    mask = image > 120
    if not np.any(mask):
        return image, np.zeros_like(image), np.zeros_like(image, dtype=bool), np.zeros_like(image, dtype=bool)
    

    # Add spatial context via neighborhood averaging
    spatial = uniform_filter(image, size=3)
    spatial_feat = 2 * spatial[mask]

   
    

    # Feature construction (x, y, intensity, gradient)
    coords = np.argwhere(mask)
    x_norm = coords[:, 1] / w
    y_norm = coords[:, 0] / h
    intensity = 5 * image[mask]
    grad = gaussian_gradient_magnitude(image, sigma=1)
    grad_feat = 2 * grad[mask]

    features = np.stack((x_norm, y_norm, intensity, grad_feat, spatial_feat), axis=0)


    # FCM clustering
    cntr, u, *_ = fuzz.cluster.cmeans(
        data=features, c=cluster_n, m=3, error=0.005, maxiter=150, seed=0
    )

    needle_cluster = np.argmax(cntr[:, 2])
    membership_flat = np.zeros(h * w)
    membership_flat[np.flatnonzero(mask)] = u[needle_cluster]
    membership = membership_flat.reshape(h, w)

    thresh = filters.threshold_otsu(membership)
    binary_mask = membership > thresh

   
    return image, membership, binary_mask



# Number of needles to keep in the final segmentation
NUM_NEEDLES = 11

from skimage.measure import label, regionprops

# segment first slice
img = image_data[:, :, 0]
original, membership, binary_mask = segment_needle_slice(img)


labels = label(binary_mask, connectivity=1)

plt.imshow(labels, cmap='nipy_spectral')
plt.show()
# props = regionprops(labels)
# props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
# keep_labels = [p.label for p in props_sorted[:NUM_NEEDLES+3]]
# output_mask = np.isin(labels, keep_labels).astype(np.uint8)

current_label = (labels == 17) 
coords = np.argwhere(current_label)
centroid = coords.mean(axis=0)
centroid = tuple(centroid)
print("Centroid:", centroid) 

cy, cx = centroid  # float
cy = int(round(cy))
cx = int(round(cx))

# Define crop boundaries with clamping to image size
y_min = max(cy - 20, 0)
y_max = min(cy + 21, image_data.shape[0])  # +21 to include 41 pixels
x_min = max(cx - 20, 0)
x_max = min(cx + 21, image_data.shape[1])

# Crop the 2D image
cropped = image_data[y_min:y_max, x_min:x_max, :]

from skimage.filters import try_all_threshold
from skimage import filters, measure, morphology

for i in range(image_data.shape[2]):
    img = cropped[:, :, i]
    
    
    thresh = filters.threshold_yen(img)
    binary = (img > thresh)
    labeled = measure.label(binary)
    props = measure.regionprops(labeled) 
    # props = [p for p in props if p.label != 0]
    

    largest = max(props, key=lambda r: r.area) # Take largest object
    area = largest.area
    perimeter = largest.perimeter
    major = largest.major_axis_length
    minor = largest.minor_axis_length
    largest_label = largest.label
    
    # Keep only largest island
    binary_largest = labeled == largest_label
    
    print(f'Ellipticity: {1 - (minor / major)}')
    print(f'Circularity: {(4 * np.pi * area) / (perimeter ** 2)}')  

    # fig, ax = try_all_threshold(img, figsize=(10, 10), verbose=False)
    # plt.show()
    plt.imshow(binary_largest, cmap='gray')
    plt.show()


exit()

# Allocate 3D mask volume
mask_3d = np.zeros_like(image_data, dtype=bool)

for i in range(image_data.shape[2]):
    img = image_data[:, :, i]
    original, membership, binary_mask = segment_needle_slice(img)

    mask_3d[:, :, i] = binary_mask

    

    # fig, axes = plt.subplots(1, 3, figsize=(10, 14))

    # axes[0].imshow(original, cmap='gray')
    # axes[0].set_title(f"Slice {i} - Original")
    # axes[0].axis('off')

    # axes[1].imshow(membership, cmap='hot')
    # axes[1].set_title("FCM Membership")
    # axes[1].axis('off')

    # axes[2].imshow(binary_mask, cmap='gray')
    # axes[2].set_title("Thresholded Mask")
    # axes[2].axis('off')

    # plt.tight_layout()
    # plt.show()

from skimage.measure import label, regionprops

# Step 1: Label components
labels = label(mask_3d)


mask_cleaned = np.zeros_like(labels, dtype=labels.dtype)
for labelIndx in np.unique(labels):
    if labelIndx == 0:
        continue  # Skip background
    
    current_label = (labels == labelIndx)    

    props = regionprops(current_label.astype(int))
    if props[0].area > 50:
        mask_cleaned[current_label] = labelIndx

# from scipy.stats import gaussian_kde
# from scipy.spatial.distance import cdist

# kdevals = []
# for i in range(image_data.shape[2]):
#     img = image_data[:, :, i]
#     mask = mask_cleaned[:, :, i].astype(bool)
#     mask_pixels = img*mask

#     pixels = mask_pixels.ravel()
#     pixels = pixels[pixels != 0]
 
#     hist,bin_edges = np.histogram(pixels,bins=256,range=(0,255))
#     bin_centers = (bin_edges[:-1]+bin_edges[1:])/2

#     kde = gaussian_kde(pixels)
#     xvals = np.linspace(bin_edges[0], bin_edges[-1],1000)
#     kde_vals = kde(xvals)
#     kdevals.append(kde_vals)

#     plt.plot(xvals,kde_vals)

#     #plt.plot(bin_edges[:-1], hist)

# kdevals = np.array(kdevals)
# dist_matrix = cdist(kdevals, kdevals, metric='euclidean')
# dist_sums = dist_matrix.sum(axis=1)

# medoid_idx = np.argmin(dist_sums)
# medoid_curve = kdevals[medoid_idx]

# plt.plot(xvals,medoid_curve,color='red',linewidth=2.5)

# plt.xlim(0,255)

# plt.show()



exit()
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from math import degrees, acos

# === START with mask_cleaned: a 3D labeled volume ===
# Assumes mask_cleaned already contains labeled needle structures (e.g., from regionprops)

# === STEP 1: Extract per-slice centroids for each component ===
needle_lines = []
for label_index in np.unique(mask_cleaned):
    if label_index == 0:
        continue
    current_label = (mask_cleaned == label_index)
    points = []
    for z in range(current_label.shape[2]):
        slice_mask = current_label[:, :, z]
        if np.any(slice_mask):
            props = regionprops(slice_mask.astype(int))
            centroid = props[0].centroid
            points.append((centroid[1], centroid[0], z))  # (x, y, z)
    if len(points) > 1:
        needle_lines.append(points)

# === STEP 2: Merge logic based on cosine angle ===

def get_direction_vector(pts):
    pts = sorted(pts, key=lambda p: p[2])  # sort by z
    p0, p1 = np.array(pts[0]), np.array(pts[-1])
    vec = p1 - p0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def angle_between_vectors(v1, v2):
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return degrees(acos(cos_sim))

def is_merge_candidate(n1, n2, angle_threshold_deg=10, xy_thresh=50):
    # Bottom of n1 must be above top of n2 (z-axis)
    top1, bottom1 = sorted(n1, key=lambda p: p[2])[0], sorted(n1, key=lambda p: p[2])[-1]
    top2, bottom2 = sorted(n2, key=lambda p: p[2])[0], sorted(n2, key=lambda p: p[2])[-1]
    if bottom1[2] >= top2[2]:
        return False
    xy_dist = np.linalg.norm(np.array(bottom1[:2]) - np.array(top2[:2]))
    if xy_dist > xy_thresh:
        return False

    # Angle check
    dir1 = get_direction_vector(n1)
    dir2 = get_direction_vector(n2)
    angle = angle_between_vectors(dir1, dir2)
    return angle <= angle_threshold_deg

def merge_needle_groups_cosine(needle_lines):
    n = len(needle_lines)
    merged_groups = []
    used = [False] * n

    for i in range(n):
        if used[i]:
            continue
        group = [i]

        for j in range(n):
            if i == j or used[j]:
                continue
            if is_merge_candidate(needle_lines[i], needle_lines[j]) or is_merge_candidate(needle_lines[j], needle_lines[i]):
                group.append(j)

        # Ensure all combinations within group are consistent
        all_good = True
        for a, b in combinations(group, 2):
            if not (is_merge_candidate(needle_lines[a], needle_lines[b]) or is_merge_candidate(needle_lines[b], needle_lines[a])):
                all_good = False
                break

        if all_good:
            merged = []
            for idx in group:
                used[idx] = True
                merged += needle_lines[idx]
            merged_groups.append(merged)

    return merged_groups

# === STEP 3: Run merging algorithm ===
merged_needles = merge_needle_groups_cosine(needle_lines)

# === STEP 4: Plot the results ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for line in merged_needles:
    line = sorted(line, key=lambda p: p[2])
    xs, ys, zs = zip(*line)
    ax.plot(xs, ys, zs, color='red', marker=None)

ax.set_title("Merged Needle Paths (Cosine Angle < 5°)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()






fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for labelIndx in np.unique(mask_cleaned):
    if labelIndx == 0:
        continue  # Skip background
    
    current_label = (labels == labelIndx)   

    points = []
    for i in range(image_data.shape[2]):
        if np.any(current_label[:, :, i]):
            props = regionprops(current_label[:, :, i].astype(int))
            points.append((props[0].centroid[1], props[0].centroid[0], i))  # (y, x, slice)
        
    if len(points) >= 2:
        xs, ys, zs = zip(*points)
        ax.plot(xs, ys, zs, color='red')  # Line + markers

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
          

     






    






for sliceIndx in range(image_data.shape[2]):
    print(f"Processing slice {sliceIndx + 1}/{image_data.shape[2]}")


    image = img_as_float(image_data[:, :, sliceIndx])
    h, w = image.shape

    # === Fuzzy C-Means segmentation ===
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_norm = xx.ravel() / w
    y_norm = yy.ravel() / h
    features = np.stack((x_norm, y_norm, 5 * image.ravel()), axis=0)

    cntr, u, *_ = fuzz.cluster.cmeans(
        data=features, c=4, m=2, error=0.005, maxiter=1000, seed=0
    )
    needle_cluster = np.argmax(cntr[:, 2])
    needle_membership = u[needle_cluster].reshape(h, w)

    plt.imshow(needle_membership, cmap='hot')
    plt.show()

#     # === Threshold segmentation ===
#     needle_mask_fuzzy = (needle_membership > 0.3) & (needle_membership < 0.6)

#     # === Label connected regions ===
#     labels = label(needle_mask_fuzzy)

#     # === Measure region properties ===
#     regions = regionprops(labels)

#     # === Sort regions by area (descending) ===
#     regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)

#     # === Keep only 14 largest regions ===
#     keep_labels = [r.label for r in regions_sorted[:14]]

#     # === Create new mask with only top 14 regions ===
#     fuzzyClusteringAreaFilterMask[:,:,sliceIndx] = np.isin(labels, keep_labels).T.astype(bool)

#     # === Extract intensities and positions from masked region ===
#     masked_coords = np.argwhere(needle_mask_fuzzy)
#     masked_intensities = image[needle_mask_fuzzy].reshape(-1, 1)

#     # === Run DBSCAN on intensities ===
#     db = DBSCAN(eps=0.02, min_samples=10).fit(masked_intensities)
#     labels = db.labels_

#     # === Label image (full) ===
#     cluster_image = np.full((h, w), -1, dtype=int)
#     for i, (y, x) in enumerate(masked_coords):
#         cluster_image[y, x] = labels[i]

#     # === Filter clusters by density ===
#     mask = cluster_image > 3

#     # === Convex Hull Fill for Each Region ===
#     label_image = label(mask)
#     convex_filled_mask = np.zeros_like(mask, dtype=bool)

#     for region in regionprops(label_image):
#         coords = region.coords
#         if coords.shape[0] >= 3:
#             try:
#                 hull = ConvexHull(coords)
#                 hull_coords = coords[hull.vertices]
#                 rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], shape=mask.shape)
#                 convex_filled_mask[rr, cc] = True
#             except:
#                 continue
   
#     fuzzyClusteringDBSCANFilterMask[:,:,sliceIndx] = convex_filled_mask.T.astype(bool)




# # Suppose these are your 3D boolean masks
# mask1 = fuzzyClusteringAreaFilterMask.astype(np.uint8)
# mask2 = fuzzyClusteringDBSCANFilterMask.astype(np.uint8)

# # Use the affine from an existing NIfTI file to preserve spatial info
# template = nib.load(r"C:\Users\arkaniva\Projects\Master-Thesis-BrachyTherapy\image2.nii")
# affine = template.affine
# header = template.header.copy()  # Optional

# # Create new NIfTI images
# img1 = nib.Nifti1Image(mask1, affine, header)
# img2 = nib.Nifti1Image(mask2, affine, header)

# # Save to disk
# nib.save(img1, "mask_area_filter.nii")
# nib.save(img2, "mask_dbscan_filter.nii")


# # === Plotting: 3x3 Grid ===
# fig, axes = plt.subplots(2, 3, figsize=(18, 18))

# # Row 1
# axes[0, 0].imshow(image, cmap='gray')
# axes[0, 0].set_title("Original Image")
# axes[0, 0].axis('off')

# axes[0, 1].imshow(needle_membership, cmap='hot')
# axes[0, 1].set_title("Needle Membership (FCM)")
# axes[0, 1].axis('off')

# axes[0, 2].imshow(filtered_mask, cmap='gray')
# axes[0, 2].set_title("Needle Mask (Thresholded)")
# axes[0, 2].axis('off')

# # Row 2
# axes[1, 0].imshow(cluster_image, cmap='nipy_spectral', vmin=-1)
# axes[1, 0].set_title("DBSCAN Clusters (Intensity Only)")
# axes[1, 0].axis('off')

# axes[1, 1].imshow(image, cmap='gray')
# axes[1, 1].imshow(mask, cmap='jet', alpha=0.3)
# axes[1, 1].set_title("High-Density DBSCAN Mask")
# axes[1, 1].axis('off')

# axes[1, 2].imshow(image, cmap='gray')
# axes[1, 2].imshow(convex_filled_mask, cmap='spring', alpha=0.2)
# axes[1, 2].set_title("Convex Hull Overlay")
# axes[1, 2].axis('off')


# plt.show()
