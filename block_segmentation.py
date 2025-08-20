```python
import nibabel as nib
import numpy as np
import skfuzzy as fuzz
from scipy.ndimage import gaussian_gradient_magnitude, uniform_filter
from skimage import filters, img_as_float
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import degrees, acos

# Set matplotlib backend to a non-interactive one if needed for servers,
# or 'TkAgg', 'Qt5Agg' etc. for interactive plots.
# import matplotlib
# matplotlib.use('TkAgg')

# --- Constants ---
# It's good practice to define constants at the top for easy configuration.
NIFTI_FILE_PATH = r"C:\Users\arkaniva\Projects\Master-Thesis-BrachyTherapy\image2.nii"
MIN_BACKGROUND_INTENSITY = 120  # Pixel intensity threshold to separate foreground
FCM_CLUSTERS = 4                # Number of clusters for Fuzzy C-Means
FCM_FUZZINESS = 3               # Fuzziness parameter 'm' for FCM
MIN_3D_OBJECT_SIZE = 50         # Minimum number of voxels for a 3D object to be kept
MERGE_ANGLE_THRESHOLD = 10.0    # Max angle (degrees) between needle segments to merge
MERGE_XY_DISTANCE_THRESHOLD = 50.0 # Max distance in XY plane to consider merging

def load_and_prepare_nifti(filepath: str) -> np.ndarray:
    """
    Loads a NIfTI file and transposes each slice for correct orientation.

    Args:
        filepath: Path to the .nii or .nii.gz file.

    Returns:
        A 3D NumPy array with the (y, x, z) axis order.
    """
    print(f"Loading NIfTI file from: {filepath}")
    nifti_file = nib.load(filepath)
    image_data = nifti_file.get_fdata()

    # Vectorized transpose is much faster than a loop.
    # This swaps the first two axes (x, y) -> (y, x) for each slice.
    return np.transpose(image_data, (1, 0, 2))


def segment_needle_slice(
    image: np.ndarray,
    n_clusters: int = FCM_CLUSTERS,
    fuzziness: float = FCM_FUZZINESS,
    background_thresh: int = MIN_BACKGROUND_INTENSITY,
) -> np.ndarray:
    """
    Segments needle-like structures in a single 2D image slice using
    Fuzzy C-Means clustering on a feature space.

    Args:
        image: A 2D NumPy array representing the image slice.
        n_clusters: The number of clusters to form.
        fuzziness: The fuzziness parameter 'm' for the FCM algorithm.
        background_thresh: Intensity value below which pixels are considered background.

    Returns:
        A 2D boolean NumPy array representing the binary segmentation mask.
    """
    # Ensure image is in floating point format for calculations.
    image_float = img_as_float(image)
    h, w = image_float.shape

    # Create a mask to process only foreground pixels, improving efficiency.
    mask = image_float > background_thresh
    if not np.any(mask):
        return np.zeros_like(image_float, dtype=bool)

    # --- Feature Engineering ---
    # We create a feature vector for each foreground pixel to improve clustering.
    coords = np.argwhere(mask)
    
    # 1. Spatial Features: Normalized coordinates (x, y).
    # This helps the clustering algorithm group pixels that are close together.
    x_norm = coords[:, 1] / w
    y_norm = coords[:, 0] / h

    # 2. Intensity Feature: The pixel's brightness.
    # Scaled to give it more weight in the clustering decision.
    intensity_feat = 5.0 * image_float[mask]

    # 3. Gradient Feature: Magnitude of the intensity gradient.
    # Helps identify edges, which are characteristic of needle boundaries.
    grad = gaussian_gradient_magnitude(image_float, sigma=1)
    grad_feat = 2.0 * grad[mask]
    
    # 4. Neighborhood Feature: Average intensity in a small neighborhood.
    # This adds spatial context and smooths the feature space.
    spatial_feat = 2.0 * uniform_filter(image_float, size=3)[mask]

    features = np.stack((x_norm, y_norm, intensity_feat, grad_feat, spatial_feat), axis=1)

    # --- Fuzzy C-Means Clustering ---
    # data must be of shape (n_features, n_samples) for skfuzzy.
    cntr, u, *_ = fuzz.cluster.cmeans(
        data=features.T,
        c=n_clusters,
        m=fuzziness,
        error=0.005,
        maxiter=150,
        seed=0,
    )

    # Assume the needle cluster has the highest average intensity.
    # The intensity feature is at index 2 in our feature vector.
    needle_cluster_idx = np.argmax(cntr[:, 2])

    # Reconstruct the 2D membership map for the needle cluster.
    membership_map = np.zeros(image_float.shape, dtype=np.float32)
    membership_map[mask] = u[needle_cluster_idx]

    # --- Thresholding ---
    # Use Otsu's method to find the optimal threshold for the membership map,
    # converting it into a binary mask.
    try:
        thresh = filters.threshold_otsu(membership_map[membership_map > 0])
        binary_mask = membership_map > thresh
    except ValueError:
        # This can happen if the membership map is all zeros.
        return np.zeros_like(image_float, dtype=bool)

    return binary_mask


def post_process_3d_mask(mask_3d: np.ndarray, min_size: int) -> np.ndarray:
    """
    Cleans a 3D binary mask by removing small, disconnected objects.

    Args:
        mask_3d: The 3D binary mask.
        min_size: The minimum number of voxels for an object to be kept.

    Returns:
        The cleaned 3D labeled mask.
    """
    # Label connected components in 3D.
    labels = label(mask_3d, connectivity=2)
    
    # Get properties of each labeled region.
    props = regionprops(labels)
    
    # Identify labels of regions that are large enough.
    labels_to_keep = {prop.label for prop in props if prop.area >= min_size}
    
    # Create the cleaned mask using efficient array indexing.
    cleaned_mask = np.isin(labels, list(labels_to_keep))
    
    # Return the final labeled volume for further processing.
    return label(cleaned_mask)


def extract_needle_centroids(labeled_mask: np.ndarray) -> list:
    """
    Extracts the per-slice centroids for each labeled object in a 3D volume.

    Args:
        labeled_mask: A 3D NumPy array where each connected component has a unique integer label.

    Returns:
        A list of "needle lines," where each line is a list of (x, y, z) centroid coordinates.
    """
    needle_lines = []
    unique_labels = np.unique(labeled_mask)
    
    for label_idx in unique_labels:
        # Skip the background label (0).
        if label_idx == 0:
            continue
            
        points = []
        # Find slices where the current label exists.
        z_indices = np.unique(np.where(labeled_mask == label_idx)[2])

        for z in z_indices:
            # Get the 2D slice for the current label.
            slice_mask = (labeled_mask[:, :, z] == label_idx)
            props = regionprops(slice_mask.astype(np.uint8))
            if props:
                # Store centroid as (x, y, z) for easier plotting.
                centroid_y, centroid_x = props[0].centroid
                points.append((centroid_x, centroid_y, z))
        
        # Only consider lines with at least two points.
        if len(points) > 1:
            needle_lines.append(points)
            
    return needle_lines


def merge_collinear_needles(
    needle_lines: list,
    angle_thresh_deg: float,
    xy_thresh: float,
) -> list:
    """
    Merges broken needle segments that are collinear and close to each other.

    Args:
        needle_lines: A list of needle segments, each a list of (x,y,z) points.
        angle_thresh_deg: The maximum angle between direction vectors to consider merging.
        xy_thresh: The maximum XY distance between the end of one segment and the start of another.

    Returns:
        A list of merged needle lines.
    """
    
    def get_direction_vector(points: list) -> np.ndarray:
        """Calculates the normalized direction vector from the first to the last point."""
        # Sort points by z-axis to ensure consistent direction.
        p_start = np.array(points[0])
        p_end = np.array(points[-1])
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else np.zeros(3)

    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates the angle in degrees between two vectors."""
        cos_sim = np.dot(v1, v2)
        # Clip to handle potential floating point inaccuracies.
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return degrees(acos(cos_sim))

    if not needle_lines:
        return []

    # Sort each individual needle line by z-coordinate.
    for line in needle_lines:
        line.sort(key=lambda p: p[2])
    
    # Create a graph where nodes are needles and edges are potential merges.
    n = len(needle_lines)
    adj = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check for merge potential in both directions (i -> j and j -> i).
            for n1_idx, n2_idx in [(i, j), (j, i)]:
                n1, n2 = needle_lines[n1_idx], needle_lines[n2_idx]
                
                # End of n1 must be below start of n2 (along z-axis).
                if n1[-1][2] >= n2[0][2]:
                    continue

                # Check XY proximity.
                dist_xy = np.linalg.norm(np.array(n1[-1][:2]) - np.array(n2[0][:2]))
                if dist_xy > xy_thresh:
                    continue

                # Check angle between direction vectors.
                dir1 = get_direction_vector(n1)
                dir2 = get_direction_vector(n2)
                angle = angle_between(dir1, dir2)

                if angle <= angle_thresh_deg:
                    adj[i].append(j)
                    adj[j].append(i)
                    break  # Found a merge condition, no need to check other direction.

    # Find connected components in the graph to form the merged groups.
    visited = [False] * n
    merged_needles = []
    for i in range(n):
        if not visited[i]:
            component_indices = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                component_indices.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            
            # Combine all points from the needles in the component.
            full_needle = []
            for idx in component_indices:
                full_needle.extend(needle_lines[idx])
            # Sort the final merged needle by z-coordinate.
            full_needle.sort(key=lambda p: p[2])
            merged_needles.append(full_needle)
            
    return merged_needles


def plot_3d_needles(ax: Axes3D, needle_lines: list, title: str, **plot_kwargs):
    """
    Plots a list of 3D needle paths on a given Matplotlib 3D axis.

    Args:
        ax: The Matplotlib 3D axis object.
        needle_lines: A list of needle paths to plot.
        title: The title for the plot.
        **plot_kwargs: Additional keyword arguments for ax.plot().
    """
    for line in needle_lines:
        # Unzip the list of (x, y, z) tuples into three separate lists.
        xs, ys, zs = zip(*line)
        ax.plot(xs, ys, zs, **plot_kwargs)

    ax.set_title(title)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate (Slice Number)')
    # Improve viewing angle and layout.
    ax.view_init(elev=20, azim=-65)


def main():
    """
    Main function to run the complete needle segmentation and merging pipeline.
    """
    # --- 1. Load and Prepare Data ---
    image_data = load_and_prepare_nifti(NIFTI_FILE_PATH)
    num_slices = image_data.shape[2]

    # --- 2. Perform Slice-wise Segmentation ---
    print("Segmenting all slices using Fuzzy C-Means...")
    # Use a list comprehension for a concise and efficient loop.
    slice_masks = [
        segment_needle_slice(image_data[:, :, i]) for i in range(num_slices)
    ]
    # Stack the 2D masks into a 3D volume.
    mask_3d = np.stack(slice_masks, axis=-1)
    
    # --- 3. Post-process 3D Mask ---
    print("Cleaning 3D mask by removing small objects...")
    cleaned_labeled_mask = post_process_3d_mask(mask_3d, MIN_3D_OBJECT_SIZE)

    # --- 4. Extract Needle Centroids ---
    print("Extracting centroids from cleaned mask...")
    initial_needles = extract_needle_centroids(cleaned_labeled_mask)
    print(f"Found {len(initial_needles)} initial needle segments.")

    # --- 5. Merge Collinear Needles ---
    print("Merging collinear needle segments...")
    merged_needles = merge_collinear_needles(
        initial_needles, MERGE_ANGLE_THRESHOLD, MERGE_XY_DISTANCE_THRESHOLD
    )
    print(f"Resulted in {len(merged_needles)} merged needles.")

    # --- 6. Visualize Results ---
    print("Generating 3D visualizations...")
    fig = plt.figure(figsize=(18, 9))

    # Plot initial, un-merged needle segments.
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_needles(
        ax1, initial_needles, "Needle Segments Before Merging", marker='o', markersize=2
    )
    
    # Plot final, merged needle paths.
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_needles(
        ax2, merged_needles, "Needle Paths After Merging", marker='.', markersize=1, linewidth=2
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```