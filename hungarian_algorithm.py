# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import img_as_float
# from skimage.morphology import remove_small_objects
# from skimage.measure import label, regionprops
# import nibabel as nib
# import skfuzzy as fuzz
# from mpl_toolkits.mplot3d import Axes3D

# # === Load NIfTI volume ===
# nifti_file = nib.load(r"C:\Users\arkaniva\Projects\Master-Thesis-BrachyTherapy\image2.nii")
# volume = nifti_file.get_fdata()
# volume = img_as_float(volume)

# h, w, num_slices = volume.shape

# # === Storage for masks ===
# needle_masks = np.zeros((h, w, num_slices), dtype=np.uint8)

# # === Step 1: Compute fuzzy needle masks for all slices ===
# for idx in range(num_slices):
#     print(f"Processing slice {idx + 1}/{num_slices}...")
#     image = volume[:, :, idx]

#     # Features: [x, y, intensity] with weighted intensity
#     yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
#     x_norm = xx.ravel() / w
#     y_norm = yy.ravel() / h
#     intensity = image.ravel()
#     features = np.stack((x_norm, y_norm, 5 * intensity), axis=0)  # 5x intensity weighting

#     # Fuzzy C-Means clustering
#     cntr, u, *_ = fuzz.cluster.cmeans(
#         data=features, c=4, m=2, error=0.005, maxiter=1000, seed=0
#     )

#     # Select cluster with highest intensity center
#     needle_lbl = np.argmax(cntr[:, 2])
#     needle_membership = u[needle_lbl].reshape(h, w)

#     # Threshold and clean up small regions
#     needle_mask = (needle_membership > 0.3) & (needle_membership < 0.6)
#     needle_mask = remove_small_objects(needle_mask, min_size=10, connectivity=1)

#     # Store mask
#     needle_masks[:, :, idx] = needle_mask.astype(np.uint8)

# # === Step 2: Connected component analysis and centroid tracking ===
# all_centroids = []

# for z in range(num_slices):
#     mask = needle_masks[:, :, z]
#     labeled = label(mask)
#     props = regionprops(labeled)

#     for prop in props:
#         y, x = prop.centroid  # row, col
#         all_centroids.append((x, y, z))  # x=col, y=row, z=slice

# all_centroids = np.array(all_centroids)

# # === Step 3: 3D plot of centroids ===
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(all_centroids[:, 0], all_centroids[:, 1], all_centroids[:, 2], 'g-', label="Needle Path")
# ax.scatter(all_centroids[:, 0], all_centroids[:, 1], all_centroids[:, 2], color='red', s=10)

# ax.set_xlabel('X (column)')
# ax.set_ylabel('Y (row)')
# ax.set_zlabel('Slice (Z)')
# ax.set_title('3D Needle Centroid Path')
# ax.legend()
# plt.tight_layout()
# plt.show()

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



# === Load image slice ===
nifti_file = nib.load(r"C:\Users\arkaniva\Projects\Master-Thesis-BrachyTherapy\image2.nii")
image_data = nifti_file.get_fdata()
vol = np.zeros_like(image_data, dtype=image_data.dtype) # Ensure float32 for processing
for i in range(image_data.shape[2]):
    vol[:, :, i] = image_data[:, :, i].T
image_data = vol


i = 0
# Initialize the plot
fig, ax = plt.subplots()
im = ax.imshow(image_data[:, :, i+1]-image_data[:, :, i], cmap='gray')
ax.set_title("Slice 0")

# Loop through z-axis slices
for i in range(1, image_data.shape[2]-1):
    im.set_data(image_data[:, :, i+1] - image_data[:, :, i])
    ax.set_title(f"Slice {i}")
    plt.pause(5)  # Let GUI update

plt.show()

fuzzyClusteringAreaFilterMask = np.zeros_like(image_data, dtype=bool)
fuzzyClusteringDBSCANFilterMask = np.zeros_like(image_data, dtype=bool)


exit()

for sliceIndx in range(image_data.shape[2]):
    print(f"Processing slice {sliceIndx + 1}/{image_data.shape[2]}")


    image = img_as_float(image_data[:, :, sliceIndx].T)
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

    # === Threshold segmentation ===
    needle_mask_fuzzy = (needle_membership > 0.3) & (needle_membership < 0.6)

    # === Label connected regions ===
    labels = label(needle_mask_fuzzy)

    # === Measure region properties ===
    regions = regionprops(labels)

    # === Sort regions by area (descending) ===
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)

    # === Keep only 14 largest regions ===
    keep_labels = [r.label for r in regions_sorted[:14]]

    # === Create new mask with only top 14 regions ===
    fuzzyClusteringAreaFilterMask[:,:,sliceIndx] = np.isin(labels, keep_labels).T.astype(bool)

    # === Extract intensities and positions from masked region ===
    masked_coords = np.argwhere(needle_mask_fuzzy)
    masked_intensities = image[needle_mask_fuzzy].reshape(-1, 1)

    # === Run DBSCAN on intensities ===
    db = DBSCAN(eps=0.02, min_samples=10).fit(masked_intensities)
    labels = db.labels_

    # === Label image (full) ===
    cluster_image = np.full((h, w), -1, dtype=int)
    for i, (y, x) in enumerate(masked_coords):
        cluster_image[y, x] = labels[i]

    # === Filter clusters by density ===
    mask = cluster_image > 3

    # === Convex Hull Fill for Each Region ===
    label_image = label(mask)
    convex_filled_mask = np.zeros_like(mask, dtype=bool)

    for region in regionprops(label_image):
        coords = region.coords
        if coords.shape[0] >= 3:
            try:
                hull = ConvexHull(coords)
                hull_coords = coords[hull.vertices]
                rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], shape=mask.shape)
                convex_filled_mask[rr, cc] = True
            except:
                continue
   
    fuzzyClusteringDBSCANFilterMask[:,:,sliceIndx] = convex_filled_mask.T.astype(bool)




# Suppose these are your 3D boolean masks
mask1 = fuzzyClusteringAreaFilterMask.astype(np.uint8)
mask2 = fuzzyClusteringDBSCANFilterMask.astype(np.uint8)

# Use the affine from an existing NIfTI file to preserve spatial info
template = nib.load(r"C:\Users\arkaniva\Projects\Master-Thesis-BrachyTherapy\image2.nii")
affine = template.affine
header = template.header.copy()  # Optional

# Create new NIfTI images
img1 = nib.Nifti1Image(mask1, affine, header)
img2 = nib.Nifti1Image(mask2, affine, header)

# Save to disk
nib.save(img1, "mask_area_filter.nii")
nib.save(img2, "mask_dbscan_filter.nii")


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
