import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, exposure
from PIL import Image

from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage.util.shape import view_as_windows
from skimage.util import montage

import matplotlib.pyplot as plt

from skimage.measure import label
from skimage import data
from skimage import color
from skimage.morphology import extrema
from skimage import exposure
from skimage import filters
from skimage.measure import label, regionprops



# Load the NIfTI file
nifti_file = nib.load(r"C:\Users\arkaniva\Projects\Master-Thesis-BrachyTherapy\image2.nii")

# Access the image data as a NumPy array
image_data = nifti_file.get_fdata()

# View shape of data
print(image_data.shape)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from skimage import filters

# Prepare the 6th slice (Python index 5)
slice_idx = 5
img = image_data[:, :, slice_idx]
img_flat = img.ravel().reshape(-1, 1)

# 1. KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(img_flat.copy()).reshape(img.shape)



# 3. DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan_labels = dbscan.fit_predict(img_flat.copy()).reshape(img.shape)
# DBSCAN may label noise as -1, so shift for display
dbscan_labels_display = np.where(dbscan_labels == -1, 0, dbscan_labels)

# 4. Spectral Clustering
threshold = filters.threshold_yen(img.copy())
mask = img.copy() > threshold
print(mask.shape)
spectral_labels = img*np.logical_not(mask).astype(int)
print(spectral_labels.shape)
# spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', random_state=0)
# spectral_labels = spectral.fit_predict(img_flat).reshape(img.shape)

# 5. MeanShift
bandwidth = estimate_bandwidth(img_flat.copy(), quantile=0.2, n_samples=500)
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_labels = meanshift.fit_predict(img_flat).reshape(img.shape)

# 6. Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=0)
gmm_labels = gmm.fit_predict(img_flat.copy()).reshape(img.shape)

groundtruth = img

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

titles = [
    "KMeans",
    "Spectral Clustering",
    "DBSCAN",
    "MeanShift",
    "GMM"
    'GT'
]
segmentations = [
    kmeans_labels,
    spectral_labels,
    dbscan_labels_display,
    meanshift_labels,
    gmm_labels,
    groundtruth
]

for ax, seg, title in zip(axes, segmentations, titles):
    ax.imshow(seg, cmap='nipy_spectral')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("Segmentation of 6th Slice by Various Clustering Methods", fontsize=16)
plt.tight_layout()
plt.show()
exit()

threshold = filters.threshold_yen(image_data)
mask = image_data > threshold

masked_pixels = image_data[mask]

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Assume masked_pixels is a 1D array of pixel values under the mask
X = masked_pixels.ravel().reshape(-1, 1)

# Fit a Gaussian Mixture Model with 2 components (change n_components as needed)
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(X)

# Predict the component for each pixel
labels = gmm.predict(X)

# Plot histogram with 256 bins
# plt.figure(figsize=(6, 4))
# plt.hist(X, bins=256, color='gray', alpha=0.5, label='Masked Pixels')

# For each GMM component, plot mean, min, and max
mins = []
for i in range(gmm.n_components):
    # Mean of the component
    mean = gmm.means_[i, 0]
    # All pixels assigned to this component
    component_pixels = X[labels == i]
    # Min and max of the component
    comp_min = component_pixels.min()
    print(f'GMM {i+1} Mean: {mean}, Min: {comp_min}')
    mins.append(comp_min)
    # comp_max = component_pixels.max()
    # # Plot
    # plt.axvline(mean, color='r', linestyle='--', label=f'GMM {i+1} Mean')
    # plt.axvline(comp_min, color='b', linestyle=':', label=f'GMM {i+1} Min')
    # plt.axvline(comp_max, color='m', linestyle=':', label=f'GMM {i+1} Max')

# plt.title('GMM on Masked Pixels')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()
# # Plot histogram
# plt.figure(figsize=(6, 4))
# plt.hist(masked_pixels.ravel(), bins=256, color='blue', alpha=0.7)
# plt.title('Histogram of Pixel Values Under Mask')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()

# exit()
print( np.max(mins))

mask = image_data > np.max(mins)  # Use the minimum of the GMM components as the threshold

mask = label(mask, connectivity=2)  # Use 2D connectivity for labeling





img1 = mask[:,:,6].T

plt.imshow(img1, cmap='jet')
plt.show()




exit()

result_1 = filters.unsharp_mask(img, radius=1, amount=1, preserve_range=True)


plt.imshow(result_1, cmap='gray')
plt.show()

fig, ax = filters.try_all_threshold(result_1,figsize=(18, 14), verbose=False)
plt.tight_layout()
plt.show()

result_2 = filters.unsharp_mask(img, radius=5, amount=2, preserve_range=True)

plt.imshow(result_2, cmap='gray')
plt.show()

fig, ax = filters.try_all_threshold(result_2,figsize=(18, 14), verbose=False)
plt.tight_layout()
plt.show()


result_3 = filters.unsharp_mask(img, radius=20, amount=5, preserve_range=True)

plt.imshow(result_3, cmap='gray')
plt.show()

fig, ax = filters.try_all_threshold(result_3,figsize=(18, 14), verbose=False)
plt.tight_layout()
plt.show()


exit()



######################################################################################

import numpy as np
from skimage import io, util
from skimage.measure import shannon_entropy
from tqdm import tqdm

def compute_entropy_kernel(image, kernel_size=10):
    # Make sure image is grayscale and of type uint8
    if len(image.shape) > 2:
        from skimage.color import rgb2gray
        image = rgb2gray(image)
    image = util.img_as_ubyte(image)

    height, width = image.shape
    entropy_image = np.zeros_like(image, dtype=np.float32)

    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='reflect')

    for i in tqdm(range(height)):
        for j in range(width):
            # Extract 10x10 kernel centered at (i, j)
            roi = padded_image[i:i + kernel_size, j:j + kernel_size]
            entropy = shannon_entropy(roi, base=2)
            entropy_image[i, j] = entropy

    return entropy_image


# image = img/255

# entropy_img = compute_entropy_kernel(image, kernel_size=10)

# # Optional: display or save result
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image)

# plt.subplot(1, 2, 2)
# plt.title("Entropy Map")
# plt.imshow(entropy_img, cmap='hot')
# plt.colorbar()
# plt.show()

smoothened_img = filters.gaussian(img, sigma=1.5)

# plt.imshow(smoothened_img, cmap='gray')
# plt.show()


# # Define parameters for Gabor filters
# frequencies = [0.2]  # fine details for needle thickness
# thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # multiple orientations

# # Apply Gabor filters at multiple orientations and accumulate response
# filtered_real_sum = np.zeros_like(img)
# for theta in thetas:
#     real, imag = filters.gabor(img, frequency=frequencies[0], theta=theta)
#     filtered_real_sum += real  # sum the real parts to enhance needle features

# # Normalize summed response for display
# filtered_real_sum = (filtered_real_sum - filtered_real_sum.min()) / (filtered_real_sum.max() - filtered_real_sum.min())

# filtered_real_sum = img*1000

filtered_real_sum = smoothened_img

plt.imshow(filtered_real_sum, cmap='gray')
plt.show()

# filtered_real_sum = smoothened_img

print(np.max(filtered_real_sum))

filtered_real_sum = exposure.rescale_intensity(filtered_real_sum)
print(np.max(filtered_real_sum))
# We find all local maxima
local_maxima = extrema.local_maxima(filtered_real_sum)
label_maxima = label(local_maxima)
overlay = color.label2rgb(
    label_maxima, filtered_real_sum, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)]
)

# We observed in the previous image, that there are many local maxima
h = 0.4
h_maxima = extrema.h_maxima(filtered_real_sum, h)
label_h_maxima = label(h_maxima)

plt.imshow(label_h_maxima)
plt.show()

overlay_h = color.label2rgb(
    label_h_maxima, filtered_real_sum, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)]
)

# a new figure with 3 subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original image')
ax[0].axis('off')

ax[1].imshow(overlay)
ax[1].set_title('Local Maxima')
ax[1].axis('off')

ax[2].imshow(overlay_h)
ax[2].set_title(f'h maxima for h = {h:.2f}')
ax[2].axis('off')
plt.show()





exit()

import matplotlib.pyplot as plt
import numpy as np
import skimage

# %%
import nibabel as nib

# Load the NIfTI file
nifti_file = nib.load(r"C:\Users\arkaniva\Downloads\pp.nii")

# Access the image data as a NumPy array
image_data = nifti_file.get_fdata()

# View shape of data
print(image_data.shape)

from PIL import Image

from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage.util.shape import view_as_windows
from skimage.util import montage

img = np.squeeze(image_data[:,:,1].T)

import matplotlib.pyplot as plt

from skimage.measure import label
from skimage import data
from skimage import color
from skimage.morphology import extrema
from skimage import exposure


# for illustration purposes, we work on a crop of the image.
x_0 = 70
y_0 = 354
width = 100
height = 100



# the rescaling is done only for visualization purpose.
# the algorithms would work identically in an unscaled version of the
# image. However, the parameter h needs to be adapted to the scale.
img = exposure.rescale_intensity(img)

# Maxima in the galaxy image are detected by mathematical morphology.
# There is no a priori constraint on the density.

# We find all local maxima
local_maxima = extrema.local_maxima(img)
label_maxima = label(local_maxima)
overlay = color.label2rgb(
    label_maxima, img, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)]
)

# We observed in the previous image, that there are many local maxima
h = 0.5
h_maxima = extrema.h_maxima(img, h)
label_h_maxima = label(h_maxima)
overlay_h = color.label2rgb(
    label_h_maxima, img, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)]
)

# a new figure with 3 subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original image')
ax[0].axis('off')

ax[1].imshow(overlay)
ax[1].set_title('Local Maxima')
ax[1].axis('off')

ax[2].imshow(overlay_h)
ax[2].set_title(f'h maxima for h = {h:.2f}')
ax[2].axis('off')
plt.show()
#img = img2/255

# patch_shape = 8, 8
# n_filters = 49

# astro = img

# # -- filterbank1 on original image
# patches1 = view_as_windows(astro, patch_shape)
# patches1 = patches1.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
# fb1, _ = kmeans2(patches1, n_filters, minit='points')
# fb1 = fb1.reshape((-1,) + patch_shape)
# fb1_montage = montage(fb1, rescale_intensity=True)

# # -- filterbank2 LGN-like image
# astro_dog = ndi.gaussian_filter(astro, 0.5) - ndi.gaussian_filter(astro, 1)
# patches2 = view_as_windows(astro_dog, patch_shape)
# patches2 = patches2.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
# fb2, _ = kmeans2(patches2, n_filters, minit='points')
# fb2 = fb2.reshape((-1,) + patch_shape)
# fb2_montage = montage(fb2, rescale_intensity=True)

# # -- plotting
# fig, axes = plt.subplots(2, 2, figsize=(7, 6))
# ax = axes.ravel()

# ax[0].imshow(astro, cmap=plt.cm.gray)
# ax[0].set_title("Image (original)")

# ax[1].imshow(fb1_montage, cmap=plt.cm.gray)
# ax[1].set_title("K-means filterbank (codebook)\non original image")

# ax[2].imshow(astro_dog>20, cmap=plt.cm.gray)
# ax[2].set_title("Image (LGN-like DoG)")

# ax[3].imshow(fb2_montage, cmap=plt.cm.gray)
# ax[3].set_title("K-means filterbank (codebook)\non LGN-like DoG image")

# for a in ax.ravel():
#     a.axis('off')

# fig.tight_layout()
# plt.show()

# # Simulated 3D image: (depth, height, width)
# volume = image_data

# # # Perform MIP along the depth axis
# # mip = np.max(volume, axis=2)  # shape: [256, 256]

# # # Convert to image and display
# # mip_image = Image.fromarray(mip)

# import matplotlib.pyplot as plt
# import numpy as np

# from skimage import data
# from skimage.util import img_as_ubyte
# from skimage.filters.rank import entropy
# from skimage.morphology import disk

# img2 = np.squeeze(image_data[:,:,1].T)
# img = img2/255

# entr_img = entropy(img, disk(5))

# fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

# img0 = ax0.imshow(img2, cmap='gray')
# ax0.set_title("Object")
# ax1.imshow(img/(entr_img/np.max(entr_img)), cmap='gray')
# ax1.set_title("Noisy image")
# ax2.imshow(entr_img, cmap='viridis')
# ax2.set_title("Local entropy")

# fig.tight_layout()
# plt.show()

exit()
from skimage import io, color

image_color = io.imread(r'output.png')

image_gray = color.rgb2gray(image_color)  # convert to grayscale float64 (0-1)
    

# %%
plt.imshow(image_gray, cmap='gray')
plt.show()
# %%
img = np.squeeze(image_data[:,:,6].T)

import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

def detect_peaks_2d(image, neighborhood_size=3, threshold=0):
    """
    Detect 2D peaks in a grayscale image.

    Args:
        image (2D np.array): input grayscale image
        neighborhood_size (int): size of the neighborhood to consider for local maxima
        threshold (float): minimum intensity to be considered a peak

    Returns:
        list of (row, col) coordinates of peaks
    """
    # Find local maxima using a maximum filter
    local_max = (maximum_filter(image, size=neighborhood_size) == image)

    # Apply threshold to filter out low intensity maxima
    detected_peaks = local_max & (image > threshold)

    # Label connected components (peaks)
    labeled, num_features = label(detected_peaks)

    # Extract peak coordinates (center of each connected component)
    slices = find_objects(labeled)
    peaks = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) // 2
        y_center = (dy.start + dy.stop - 1) // 2
        peaks.append((y_center, x_center))

    return peaks

from skimage import io, filters, exposure

image_eq = img/255

image_eq = exposure.equalize_adapthist(image_eq)

# Apply Gaussian smoothing
image_eq = filters.gaussian(image_eq, sigma=1.0)  # try sigma=0.5 to 3.0 depending on image


# Define parameters for Gabor filters
frequencies = [0.2]  # fine details for needle thickness
thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # multiple orientations

# Apply Gabor filters at multiple orientations and accumulate response
filtered_real_sum = np.zeros_like(image_eq)
for theta in thetas:
    real, imag = filters.gabor(image_eq, frequency=frequencies[0], theta=theta)
    filtered_real_sum += real  # sum the real parts to enhance needle features

# Normalize summed response for display
filtered_real_sum = (filtered_real_sum - filtered_real_sum.min()) / (filtered_real_sum.max() - filtered_real_sum.min())

# Detect peaks with neighborhood size 3 and threshold 5
peaks = detect_peaks_2d(filtered_real_sum, neighborhood_size=3, threshold=5)
print("Detected peaks:", peaks)

# Prepare grid for 3D plotting
x = np.arange(filtered_real_sum.shape[1])
y = np.arange(filtered_real_sum.shape[0])
X, Y = np.meshgrid(x, y)

# Plot 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, filtered_real_sum, cmap='viridis', alpha=0.8)

# Mark peaks with red dots
for (py, px) in peaks:
    ax.scatter(px, py, img[py, px], color='red', s=50)

ax.set_xlabel('X (column)')
ax.set_ylabel('Y (row)')
ax.set_zlabel('Intensity (Z)')
ax.set_title('3D Surface Plot of Image with Detected Peaks')
plt.show()
