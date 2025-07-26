import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt, binary_dilation, convolve
from collections import defaultdict
from skimage import data, filters,morphology
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def signed_distance(mask):
    mask = (mask > 0).astype(np.uint8)
    dist_out = distance_transform_edt(mask == 0)
    dist_in = distance_transform_edt(mask == 1)
    return dist_in - dist_out



def filter_collinear_points(points):
    """
    From a set of (x,y) points, keep only those points lying in
    vertical or horizontal lines with at least 8 points.
    """
    x_groups = defaultdict(list)
    y_groups = defaultdict(list)

    for x, y in points:
        x_groups[x].append((x, y))
        y_groups[y].append((x, y))

    result = set()
    for group in x_groups.values():
        if len(group) > 8:
            result.update(group)
    for group in y_groups.values():
        if len(group) > 8:
            result.update(group)

    if result:
        return np.array(list(result))
    else:
        return np.zeros((0, 2), dtype=int)
    



def extractData(InputDicomFolder, OutputLocation):
    """
    Load DICOM series from InputDicomFolder, process pixel volume by:
    - bin intensities
    - detect “plus” grid markers, remove them (mark neighbors)
    - fill marked pixels by iterative 3×3 neighbor averaging
    - (other processing, saving, etc.)
    """
    
    # 1. Load DICOM files
    image_slices_filenames = glob.glob(os.path.join(InputDicomFolder, "US*"))
    files = [(pydicom.dcmread(filename), filename) for filename in image_slices_filenames]

    # Sort by slice position (assumed in ImagePositionPatient[2])
    files.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))
    dicom_files = [item[0] for item in files]
    sorted_filenames = [item[1] for item in files]

    # Metadata (orientation, affine) – kept as before
    z_positions = [float(ds.ImagePositionPatient[2]) for ds in dicom_files]
    x_cosine = np.array(dicom_files[0].ImageOrientationPatient[0:3])
    y_cosine = np.array(dicom_files[0].ImageOrientationPatient[3:6])
    z_cosine = np.cross(x_cosine, y_cosine)


    # Origin in patient coordinates
    # Note: ImagePositionPatient is in LPS (Left-Posterior-Superior) coordinate system
    # We will convert to RAS (Right-Anterior-Superior) later
    origin = [float(val) for val in dicom_files[0].ImagePositionPatient]
    affine = np.eye(4)
    affine[:3, 0] = x_cosine * dicom_files[0].PixelSpacing[0]
    affine[:3, 1] = y_cosine * dicom_files[0].PixelSpacing[1]
    affine[:3, 2] = z_cosine * dicom_files[0].SliceThickness
    affine[:3, 3] = dicom_files[0].ImagePositionPatient

    
    # Convert LPS→RAS
    lps_to_ras = np.diag([-1, -1, 1, 1])
    affine_ras = lps_to_ras @ affine

    # 2. Stack pixel arrays
    pixel_arrays = [ds.pixel_array for ds in dicom_files]

    # Cast to float32 early so that averaging works without repeated casting
    pixel_matrix = np.stack(pixel_arrays, axis=0).astype(np.float32)  # shape [Z, Y, X]
    original = pixel_matrix.copy()

    
    
    '''
    Denoise hand-drawn contour by binning pixel intensities.
    '''
    bin_size = 5
    pixel_matrix_binned = (pixel_matrix.astype(np.int32) // bin_size) * bin_size
    pixel_matrix[pixel_matrix_binned == 255] = -1  # Mark hand-drawn prostate contour: pixels equal to 255 after binning



    '''
    Detect Grids burned into the images.
        - Grids are like plus shaped pixels. If the center is detected, 
          you can extract the grid using 4-neighbourhood.
    '''

    for slice_idx in range(pixel_matrix.shape[0]):

        current_binned = pixel_matrix_binned[slice_idx] 

        center = current_binned[1:-1, 1:-1]
        up = current_binned[:-2, 1:-1]
        down = current_binned[2:, 1:-1]
        left = current_binned[1:-1, :-2]
        right = current_binned[1:-1, 2:]
        mask_center = (center == up) & (center == down) & (center == left) & (center == right) & (center >= 200) & (center <= 255)
        
        centers = np.argwhere(mask_center)
        if centers.size == 0:
            continue


        grid_unit_centers = np.stack([centers[:, 1] + 1, centers[:, 0] + 1], axis=1) 

        # Build a boolean mask of centers to dilate. We'll mark neighbors of these points.
        center_mask = np.zeros((pixel_matrix.shape[1], pixel_matrix.shape[2]), dtype=bool)
        center_mask[grid_unit_centers[:, 1], grid_unit_centers[:, 0]] = True
        dilated = binary_dilation(center_mask, structure=np.ones((3, 3), dtype=bool)) # Dilate by 3x3 to mark neighbors including diagonals
        pixel_matrix[slice_idx][dilated] = -1 # Mark these positions in pixel_matrix as -1
        pixel_matrix_copy = pixel_matrix[slice_idx].copy()  # Copy for visualization


        # Filter only collinear points (to reduce noise as non-grid pixels can be selected but grids are always collinear)
        grid_unit_centers = filter_collinear_points(grid_unit_centers)
        if grid_unit_centers.size == 0:
            continue

        
        # Compute convex hull of the grid unit centers to create a polygon around the grid
        # This will help in cropping the image around the grid
        hull = ConvexHull(grid_unit_centers)
        hull_points = grid_unit_centers[hull.vertices]

        # Close the hull for plotting (testing)
        hull_points = np.vstack([hull_points, hull_points[0]])

        # Create two copies of the current slice for processing
        # One for grid area and another for n pixels more than grid to include the alpha-numeric characters
        # current_slice_copy1 = pixel_matrix[slice_idx].copy()
        # current_slice_copy2 = pixel_matrix[slice_idx].copy()
        current_slice_copy1 = current_binned.copy()
        current_slice_copy2 = current_binned.copy()

        # plt.figure(figsize=(6, 6))
        # plt.imshow(current_slice_copy1, cmap='gray')
        # plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2, label='Convex Hull')
        # plt.scatter(grid_unit_centers[:, 0], grid_unit_centers[:, 1], c='b', s=20, label='Plus Points')
        # plt.title(f'Convex Hull of Plus Points on Slice {slice_idx}')
        # plt.show()

        # Calculate min-max of the hull points to create a bounding box
        min_x = int(np.floor(np.min(hull_points[:, 0])))
        max_x = int(np.ceil(np.max(hull_points[:, 0])))
        min_y = int(np.floor(np.min(hull_points[:, 1])))
        max_y = int(np.ceil(np.max(hull_points[:, 1])))

        # Set the outside of the bounding box to white (255) for the first copy
        current_slice_copy1[:min_y, :] = 255
        current_slice_copy1[max_y:, :] = 255
        current_slice_copy1[min_y:max_y, :min_x] = 255
        current_slice_copy1[min_y:max_y, max_x:] = 255




        # Extend the hull points by 20 pixels in each direction to include the alpha-numeric characters
        idx = np.where(hull_points[:, 0] == hull_points[:, 0].min()) # for xmin
        hull_points[idx,0] = hull_points[idx,0] - 25

        idx = np.where(hull_points[:, 0] == hull_points[:, 0].max()) # for xmax
        hull_points[idx,0] = hull_points[idx,0] + 25

        idx = np.where(hull_points[:, 1] == hull_points[:, 1].min()) # for ymin
        hull_points[idx,1] = hull_points[idx,1] - 35

        idx = np.where(hull_points[:, 1] == hull_points[:, 1].max()) # for ymax
        hull_points[idx,1] = current_slice_copy1.shape[0] # lower boundary till the end of the image to include rectum


        # plt.figure(figsize=(6, 6))
        # plt.imshow(current_slice_copy2, cmap='gray')
        # plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2, label='Convex Hull')
        # plt.scatter(grid_unit_centers[:, 0], grid_unit_centers[:, 1], c='b', s=20, label='Plus Points')
        # plt.title(f'Convex Hull of Plus Points on Slice {slice_idx}')
        # plt.show()

        # Calculate min-max of the extended hull points to create a bounding box
        min_x2 = int(np.floor(np.min(hull_points[:, 0])))
        max_x2 = int(np.ceil(np.max(hull_points[:, 0])))
        min_y2 = int(np.floor(np.min(hull_points[:, 1])))
        max_y2 = int(np.ceil(np.max(hull_points[:, 1])))

        
        # Set the outside of the bounding box to white (255) for the second copy
        current_slice_copy2[:min_y2, :] = 255
        current_slice_copy2[max_y2:, :] = 255
        current_slice_copy2[min_y2:max_y2, :min_x2] = 255
        current_slice_copy2[min_y2:max_y2, max_x2:] = 255

    
        # Subtract the two copies to get the difference image which includes the alpha-numeric characters
        subtract = current_slice_copy2 - current_slice_copy1

        # plt.figure(figsize=(6, 6))
        # plt.imshow(subtract, cmap='gray')
        # plt.title(f'Convex Hull of Plus Points on Slice {slice_idx}')
        # plt.show()



        
        # plt.figure(figsize=(6, 6))
        # plt.imshow(subtract, cmap='gray')
        # plt.show()



        # Threshold the alpha-numeric region using Otsu's method
        threshold = filters.threshold_otsu(subtract)        
        binary_image = subtract > threshold # Apply the threshold

        # plt.figure(figsize=(6, 6))
        # plt.imshow(binary_image, cmap='gray')
        # plt.show()


        binary_image = morphology.binary_dilation(binary_image)  # Dilate to fill small gaps

        # plt.figure(figsize=(6, 6))
        # plt.imshow(binary_image, cmap='gray')
        # plt.show()



        # Get coordinates of nonzero pixels in (row, col) order
        binary_coords = np.column_stack(np.where(binary_image))
        binary_coords = binary_coords[:, [1, 0]]

        # Mark foreground pixels of the alpha-numeric region 
        all_plus_centers = np.vstack([grid_unit_centers, binary_coords])
        rows = all_plus_centers[:, 1]
        cols = all_plus_centers[:, 0]
        pixel_matrix[slice_idx][rows, cols] = -1


        # Now reset the cropped area in the pixel matrix to the original pixel values
        # The grid and alpha-numeric characters from the image are marked as -1 which will be replaced 
        # by the average filtering later
        pixel_matrix[slice_idx][min_y:max_y, min_x:max_x] = pixel_matrix_copy[min_y:max_y, min_x:max_x] 
        pixel_matrix[slice_idx][:min_y2, :] = pixel_matrix_copy[:min_y2, :]
        pixel_matrix[slice_idx][max_y2:, :] = pixel_matrix_copy[max_y2:, :]
        pixel_matrix[slice_idx][min_y2:max_y2, :min_x2] = pixel_matrix_copy[min_y2:max_y2, :min_x2]
        pixel_matrix[slice_idx][min_y2:max_y2, max_x2:] = pixel_matrix_copy[min_y2:max_y2, max_x2:]
    
        
    '''
    Mean filtering to denoise the marked pixels.
    - For each slice, perform 3 iterations of 3x3 averaging for pixels marked as -1.
    '''

    # 6. Iterative 3×3 averaging (“fill”) for marked pixels
    # For each slice, do 3 iterations: for all pixels == -1, compute average of non-(-1) neighbors.
    for slice_idx in range(pixel_matrix.shape[0]):
        slice_img = pixel_matrix[slice_idx]
        if not np.any(slice_img == -1):
            continue

        # Perform 3 iterations of mean filtering
        for _ in range(3):
            bad_mask = (slice_img == -1)
            if not np.any(bad_mask):
                break

            valid_mask = ~bad_mask

            # sum of neighbors (including center if valid) via convolution
            # For invalid positions, we set value to zero in the convolution input, but since valid_mask excludes them in count, zeros don't skew the average
            sum_neighbors = convolve(np.where(valid_mask, slice_img, 0.0), np.ones((3, 3), dtype=np.float32), mode='constant', cval=0.0)
            count_neighbors = convolve(valid_mask.astype(np.int32), np.ones((3, 3), dtype=np.float32), mode='constant', cval=0)

            # Avoid division by zero: only update positions where bad_mask is True and count_neighbors > 0
            update_positions = bad_mask & (count_neighbors > 0)
            
            # Compute average only for these positions
            slice_img[update_positions] = sum_neighbors[update_positions] / count_neighbors[update_positions]

        # Assign back
        pixel_matrix[slice_idx] = slice_img
    


    '''    Compute PSNR and SSIM between the original and processed images.
    - PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) are computed
    - These metrics help evaluate the quality of the denoised images.'''
    # Load images as float (important for SSIM)
    reference = original.astype(np.uint8)
    denoised = pixel_matrix.astype(np.uint8)

    psnr_value = peak_signal_noise_ratio(reference, denoised) # Compute PSNR
    ssim_value = structural_similarity(reference, denoised) # Compute SSIM

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    


    # Transpose array to [X, Y, Z] for saving as NifTI using nibabel
    # this allows for correct orientation in the NIfTI file to visualize in 3D slicer with mask overlay
    pixel_matrix_ras = np.transpose(pixel_matrix, (2, 1, 0))  # [Z, Y, X] → [X, Y, Z]


    # Save as NIfTI
    nifti_image = nib.Nifti1Image(pixel_matrix_ras.astype(np.uint8), affine_ras)
    nib.save(nifti_image, 'image2.nii')
    # nib.save(nifti_image, f'{OutputLocation}\\image.nii')
    
    # return
    
    
    # with open(r'C:\Users\sarkaraa\Downloads\roi_mapping.json', "r") as f:
    #     roi_dict = json.load(f)

    # # Example: print the dictionary or use it
    

    # # Load RT-STRUCT DICOM file to extract anatomy contours
    # try:
    #     rt_file = glob.glob(os.path.join(InputDicomFolder, "RS*"))[0]
    #     rt_struct = pydicom.dcmread(rt_file)
    # except:
    #     # print(f'RT Struct not found: {dicom_files[0].PatientName} - {InputDicomFolder.split('\\')[-1]}')
    #     return



    # # Get ROI contours
    # for i,roi in enumerate(rt_struct.StructureSetROISequence):
    #     roi_name = roi.ROIName
        
    #     if roi_name not in roi_dict.keys():
    #         print(roi_name)
    #         print(f'ROI name not matched: {InputDicomFolder}')
        
    #     break
    #     contours =  rt_struct.ROIContourSequence[i].ContourSequence

    #     # Pre-allocate memory for binary mask
    #     mask_volume = np.zeros((pixel_matrix.shape), dtype=np.uint8)

    #     segment_slice_indexes = [] #store slice numbers to check for in-between missing segments for interpolation

    #     # Extract contour, make a closed 2D polygon, and fill the polygon to create binary mask
    #     for contour in contours:            
    #         data = np.array(contour.ContourData).reshape(-1, 3)
    #         z_coord = data[0, 2]
            
    #         # Compute slice (image coordinate) in which the contour is present from patient coordinate system
    #         z_index = np.argmin(np.abs(np.array(z_positions) - z_coord)) 
    #         segment_slice_indexes.append(z_index)

            
    #         # Calculate x,y positions of the contour points in image coordinates
    #         pixel_coords = np.round((data[:, :2] - origin[:2]) / dicom_files[0].PixelSpacing[:2]).astype(int)
    #         x_coords, y_coords = pixel_coords[:, 0], pixel_coords[:, 1]

    #         # Close the contour by stitching first and last points of the polygon
    #         x_coords = np.append(x_coords, x_coords[0])  # 
    #         y_coords = np.append(y_coords, y_coords[0])  # 

    #         # Failsafe to clip coordinates in case it is out of the image slice dimension due to float precision
    #         x_coords = np.clip(x_coords, 0, pixel_matrix_ras.shape[0] - 1)
    #         y_coords = np.clip(y_coords, 0, pixel_matrix_ras.shape[1] - 1)

    #         # Create the polygon
    #         polygon = np.column_stack((y_coords,x_coords))

    #         # Create binary mask from polygon
    #         mask = polygon2mask((pixel_matrix_ras.shape[1],pixel_matrix_ras.shape[0]), polygon)

            
    #         # Replace empty slice with calculated mask
    #         mask_volume[z_index] = mask

    #     sorted_segment_slice_indexes = sorted(segment_slice_indexes)
    #     segment_range = set(range(sorted_segment_slice_indexes[0], sorted_segment_slice_indexes[-1] + 1))
    #     missing_slices = list(segment_range - set(sorted_segment_slice_indexes))

    #     if len(missing_slices)!=0:
    #         for i in range(len(sorted_segment_slice_indexes)-1):
    #             slice_number_gap = sorted_segment_slice_indexes[i+1]-sorted_segment_slice_indexes[i]
    #             if slice_number_gap == 1:
    #                 continue
    #             for j in range(1, slice_number_gap):
    #                 alpha = i / (slice_number_gap + 1)
    #                 sdf1 = signed_distance(mask_volume[sorted_segment_slice_indexes[i]])
    #                 sdf2 = signed_distance(mask_volume[sorted_segment_slice_indexes[i+1]])

    #                 # Interpolate
    #                 sdf_interp = (1 - alpha) * sdf1 + alpha * sdf2

    #                 mask_volume[sorted_segment_slice_indexes[i]+j]  = (sdf_interp > 0).astype(np.uint8)
                   

       
    #     mask_volume = np.transpose(mask_volume, (2, 1, 0))

    #     # Save masks as NIFTI files
    #     nifti_img = nib.Nifti1Image(mask_volume, affine_ras)
    #     # roi_name = roi_name.replace('/', '-').replace('\\', '-').replace('?', '-')
    #     # nib.save(nifti_img,f'{OutputLocation}\\{roi_name}.nii')
        
    # return

    # Load DICOM file with needle coordinates

    try:
        needle_path_file = glob.glob(os.path.join(InputDicomFolder, "RP*.dcm"))[0]
        ds = pydicom.dcmread(needle_path_file)
    except:
        # print(f'Needle DICOM could not be read: {dicom_files[0].PatientName} - {InputDicomFolder.split('\\')[-1]}')
        return
    
   
    catheter_reconstructed = []  # Stores individual catheter paths
   
    private_tag = ds[0x300f, 0x1000].value
    structures = private_tag[0][0x3006, 0x0039].value

    k=0
    for struct in structures:
        contour_seq = struct.ContourSequence
        if len(contour_seq) == 1:
            contour_data = contour_seq[0].ContourData
            points = []

            

            # Extract 3D points in order: x, z, -y (flipping Y-axis)
            for i in range(0, len(contour_data), 3):
                point = [float(contour_data[i]), float(contour_data[i + 1]), float(contour_data[i + 2])]
                points.append(point)
                


            catheter_reconstructed.append(points)
            
                            
            points=np.array(points)
            print(points)
            print('-------------------')
            start = points[0]
            end = points[1]
            points = np.array([list(start + (end - start) * t) for t in np.linspace(0, 1, 16)])
            # print(points)
        
        
  #ARKA

    print(f"Number of reconstructed catheters: {len(catheter_reconstructed)}")
    
    # for struct in value:
    #     print(1)
    #     contour_seq = struct.ROIContourSequence
        
    #     for c in contour_seq:
    #         print(len(c.ContourSequence))
    #         print('ooooooooooooooooooooooooo')
        #print(contour_seq)
    
    # private_tag = ds[0x300f, 0x1000].value
    # structures = ds[0x3006, 0x0082].value
    
    # print(private_tag)
    # print(structures)
    
    return 


    num_needles = len(ds.ApplicationSetupSequence[0].ChannelSequence)

   
    # Extract all needle coordinates for each slice and convert to image coordinate system 
    needle_coordinates = {f'Needle {i}': [] for i in range(1,num_needles+1)}
   
    for i in range(num_needles):
        try:
            points = [controlPoints.ControlPoint3DPosition for controlPoints in ds.ApplicationSetupSequence[0].ChannelSequence[i].BrachyControlPointSequence]
        except:
            # print(f'Needle Coordinate could not be read: {dicom_files[0].PatientName} - {InputDicomFolder.split('\\')[-1]} - Needle {i+1}')

            continue

        points = np.array(points)

        current_needle_coordinates = np.zeros(points.shape, dtype=np.int16)

        # Convert coordinates from patient coordinate system to image coordinate system
        for j in range(points.shape[0]):
            z_index = np.argmin(np.abs(np.array(z_positions) - points[j,2]))
            pixel_coords = np.round((points[j,:2] - origin[:2]) / dicom_files[0].PixelSpacing[:2]).astype(int)
            current_needle_coordinates[j,0], current_needle_coordinates[j,1], current_needle_coordinates[j,2] = map(int, np.append(pixel_coords,z_index))
                    
        needle_coordinates[f'Needle {i+1}'] = current_needle_coordinates.tolist()
    
    # Save Needle Coordinates in a JSON file
    with open(f'{OutputLocation}\\needle_coordinates.json','w') as file:
        json.dump(needle_coordinates, file, indent=4)
            


if __name__ == '__main__':

    SeriesUIDFolders = r"C:\Users\arkaniva\Downloads\Testcase\ef"
    OutputFolder = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Transformed Data"

    import time
    start_time = time.time()

    extractData(SeriesUIDFolders, OutputFolder)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


    # SeriesUIDFolders = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Filtered Data"
    # OutputFolder = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Transformed Data"


    # import time
    # start_time = time.time()

    # # i = 0
    # for seriesFolder in os.listdir(SeriesUIDFolders):
    #     # i+=1
    #     # if i<333:
    #     #     continue
        
    #     # Create series folder if not already present
    #     if not os.path.exists(f'{OutputFolder}\\{seriesFolder}'):
    #         os.makedirs(f'{OutputFolder}\\{seriesFolder}') # Create folder with SeriesUID

        
    #     extractData(f'{SeriesUIDFolders}\\{seriesFolder}', f'{OutputFolder}\\{seriesFolder}')
        
    #     break
    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time:.6f} seconds")
    







# # Helper codes
# # # --- Visualization ---

# # # print(z_index)

# # plt.imshow(mask_volume[z_index], cmap='Reds', alpha=0.5)
# # plt.show()

# # plt.imshow(pixel_matrix[z_index], cmap='gray')
# # plt.imshow(mask_volume[z_index], cmap='Reds', alpha=0.5)
# # plt.show()

# # # Initial plot
# # fig, ax = plt.subplots()
# # img1 = ax.imshow(pixel_matrix[1], cmap='gray')
# # img2 = ax.imshow(mask_volume[1], cmap='Reds', alpha=0.5)
# # title = ax.set_title("Slice 0 with Contour Overlay")
# # ax.axis('off')

# # # Initial render and pause
# # plt.pause(5)  # Render and hold the first frame for 5 seconds

# # # Update plots in-place
# # for i in range(1, pixel_matrix.shape[0]):
# #     img1.set_data(pixel_matrix[i])
# #     img2.set_data(mask_volume[i])
# #     title.set_text(f"Slice {i} with Contour Overlay")
# #     plt.pause(1)  # Pause 1 second and refresh

# # plt.close(fig)  # Close figure when done
    


