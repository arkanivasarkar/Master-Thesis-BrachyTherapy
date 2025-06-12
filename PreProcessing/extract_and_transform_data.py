import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from scipy.ndimage import distance_transform_edt, binary_dilation, convolve
from collections import defaultdict

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
        # Return empty array of shape (0,2)
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

    # 3. Bin intensities in factors of 5
    bin_size = 5
    # Create a binned version for detection, but keep pixel_matrix float for processing
    pixel_matrix_binned = (pixel_matrix.astype(np.int32) // bin_size) * bin_size

    # 4. Mark hand-drawn prostate contour: pixels equal to 255 after binning
    mask_contour = (pixel_matrix_binned == 255)
    pixel_matrix[mask_contour] = -1  # mark as “bad”

    # Precompute kernels for later use
    # Kernel for convolution-based averaging: 3x3 of ones
    avg_kernel = np.ones((3, 3), dtype=np.float32)

    # Structure element for dilation: 3x3 full ones
    dilation_structure = np.ones((3, 3), dtype=bool)

    # 5. Iterate over slices to detect plus-shaped grid markers and mark neighbors
    num_slices, H, W = pixel_matrix.shape
    print("Starting plus-shaped marker detection and removal...")
    for slice_idx in range(num_slices):
        current_binned = pixel_matrix_binned[slice_idx]  # int array
        # We want to detect at positions where central pixel and its up/down/left/right
        # are equal and in [200,255].
        # Use slicing to compute boolean mask for centers:
        # Define:
        #   center = current_binned[1:-1, 1:-1]
        #   up = current_binned[:-2, 1:-1], down = current_binned[2:, 1:-1]
        #   left = current_binned[1:-1, :-2], right = current_binned[1:-1, 2:]
        # Then mask_center = (center == up) & (center == down) & (center == left) & (center == right) & (center >= 200) & (center <= 255)
        if H < 3 or W < 3:
            # too small to detect plus shapes
            continue
        center = current_binned[1:-1, 1:-1]
        up = current_binned[:-2, 1:-1]
        down = current_binned[2:, 1:-1]
        left = current_binned[1:-1, :-2]
        right = current_binned[1:-1, 2:]
        mask_center = (center == up) & (center == down) & (center == left) & (center == right) & (center >= 200) & (center <= 255)
        # mask_center is shape (H-2, W-2). Get coordinates of True
        centers = np.argwhere(mask_center)
        # Convert to full-coord (x,y): note argwhere gives [i,j] on mask_center; full coords are (i+1,j+1)
        if centers.size == 0:
            continue
        # Build array of (x,y) points for filter_collinear_points: x is column index, y is row index
        plus_points = np.stack([centers[:, 1] + 1, centers[:, 0] + 1], axis=1)  # shape (N,2)
        # Filter only collinear sets
        plus_points = filter_collinear_points(plus_points)
        if plus_points.size == 0:
            continue
        # Build a boolean mask of centers to dilate. We'll mark neighbors of these points.
        # Initialize empty mask of shape HxW
        center_mask = np.zeros((H, W), dtype=bool)
        # Mark centers:
        # plus_points rows: [x, y] pairs: x is col, y is row
        # So center_mask[y, x] = True
        center_mask[plus_points[:, 1], plus_points[:, 0]] = True
        # Dilate by 3x3 to mark neighbors including diagonals
        dilated = binary_dilation(center_mask, structure=dilation_structure)
        # Mark these positions in pixel_matrix as -1
        pixel_matrix[slice_idx][dilated] = -1

    # 6. Iterative 3×3 averaging (“fill”) for marked pixels
    # For each slice, do 3 iterations: for all pixels == -1, compute average of non-(-1) neighbors.
    print("Starting fill by 3×3 averaging...")
    for slice_idx in range(num_slices):
        slice_img = pixel_matrix[slice_idx]
        # If no -1 in slice, skip
        if not np.any(slice_img == -1):
            continue
        for it in range(3):
            bad_mask = (slice_img == -1)
            if not np.any(bad_mask):
                break
            valid_mask = ~bad_mask
            # sum of neighbors (including center if valid) via convolution
            # For invalid positions, we set value to zero in the convolution input, but since valid_mask excludes them in count, zeros don't skew the average
            sum_neighbors = convolve(np.where(valid_mask, slice_img, 0.0), avg_kernel, mode='constant', cval=0.0)
            count_neighbors = convolve(valid_mask.astype(np.int32), avg_kernel, mode='constant', cval=0)
            # Avoid division by zero: only update positions where bad_mask is True and count_neighbors > 0
            update_positions = bad_mask & (count_neighbors > 0)
            # Compute average only for these positions
            slice_img[update_positions] = sum_neighbors[update_positions] / count_neighbors[update_positions]
            # After assignment, some previously bad pixels become valid next iteration
        # Assign back
        pixel_matrix[slice_idx] = slice_img
    

           
       
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
   
    # Load images as float (important for SSIM)
    img1 = original.astype(np.uint8)
    img2 = pixel_matrix.astype(np.uint8)
    
    # plt.imshow(img1[10,:,:],cmap='gray')
    # plt.show()
    # plt.imshow(img2[10,:,:],cmap='gray')
    # plt.show()

    # Compute PSNR
    psnr_value = peak_signal_noise_ratio(img1, img2)

    # Compute SSIM
    ssim_value = structural_similarity(img1, img2)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    
    # # Transpose array to [X, Y, Z] for saving as NifTI using nibabel
    pixel_matrix_ras = np.transpose(original, (2, 1, 0))  # [Z, Y, X] → [X, Y, Z]

    # Save as NIfTI
    nifti_image = nib.Nifti1Image(pixel_matrix_ras, affine_ras)
    nib.save(nifti_image, 'image.nii')
    # nib.save(nifti_image, f'{OutputLocation}\\image.nii')
    
    return
    
    
    with open(r'C:\Users\sarkaraa\Downloads\roi_mapping.json', "r") as f:
        roi_dict = json.load(f)

    # Example: print the dictionary or use it
    

    # Load RT-STRUCT DICOM file to extract anatomy contours
    try:
        rt_file = glob.glob(os.path.join(InputDicomFolder, "RS*"))[0]
        rt_struct = pydicom.dcmread(rt_file)
    except:
        print(f'RT Struct not found: {dicom_files[0].PatientName} - {InputDicomFolder.split('\\')[-1]}')
        return



    # Get ROI contours
    for i,roi in enumerate(rt_struct.StructureSetROISequence):
        roi_name = roi.ROIName
        
        if roi_name not in roi_dict.keys():
            print(roi_name)
            print(f'ROI name not matched: {InputDicomFolder}')
        
        break
        contours =  rt_struct.ROIContourSequence[i].ContourSequence

        # Pre-allocate memory for binary mask
        mask_volume = np.zeros((pixel_matrix.shape), dtype=np.uint8)

        segment_slice_indexes = [] #store slice numbers to check for in-between missing segments for interpolation

        # Extract contour, make a closed 2D polygon, and fill the polygon to create binary mask
        for contour in contours:            
            data = np.array(contour.ContourData).reshape(-1, 3)
            z_coord = data[0, 2]
            
            # Compute slice (image coordinate) in which the contour is present from patient coordinate system
            z_index = np.argmin(np.abs(np.array(z_positions) - z_coord)) 
            segment_slice_indexes.append(z_index)

            
            # Calculate x,y positions of the contour points in image coordinates
            pixel_coords = np.round((data[:, :2] - origin[:2]) / dicom_files[0].PixelSpacing[:2]).astype(int)
            x_coords, y_coords = pixel_coords[:, 0], pixel_coords[:, 1]

            # Close the contour by stitching first and last points of the polygon
            x_coords = np.append(x_coords, x_coords[0])  # 
            y_coords = np.append(y_coords, y_coords[0])  # 

            # Failsafe to clip coordinates in case it is out of the image slice dimension due to float precision
            x_coords = np.clip(x_coords, 0, pixel_matrix_ras.shape[0] - 1)
            y_coords = np.clip(y_coords, 0, pixel_matrix_ras.shape[1] - 1)

            # Create the polygon
            polygon = np.column_stack((y_coords,x_coords))

            # Create binary mask from polygon
            mask = polygon2mask((pixel_matrix_ras.shape[1],pixel_matrix_ras.shape[0]), polygon)

            
            # Replace empty slice with calculated mask
            mask_volume[z_index] = mask

        sorted_segment_slice_indexes = sorted(segment_slice_indexes)
        segment_range = set(range(sorted_segment_slice_indexes[0], sorted_segment_slice_indexes[-1] + 1))
        missing_slices = list(segment_range - set(sorted_segment_slice_indexes))

        if len(missing_slices)!=0:
            for i in range(len(sorted_segment_slice_indexes)-1):
                slice_number_gap = sorted_segment_slice_indexes[i+1]-sorted_segment_slice_indexes[i]
                if slice_number_gap == 1:
                    continue
                for j in range(1, slice_number_gap):
                    alpha = i / (slice_number_gap + 1)
                    sdf1 = signed_distance(mask_volume[sorted_segment_slice_indexes[i]])
                    sdf2 = signed_distance(mask_volume[sorted_segment_slice_indexes[i+1]])

                    # Interpolate
                    sdf_interp = (1 - alpha) * sdf1 + alpha * sdf2

                    mask_volume[sorted_segment_slice_indexes[i]+j]  = (sdf_interp > 0).astype(np.uint8)
                   

       
        mask_volume = np.transpose(mask_volume, (2, 1, 0))

        # Save masks as NIFTI files
        nifti_img = nib.Nifti1Image(mask_volume, affine_ras)
        # roi_name = roi_name.replace('/', '-').replace('\\', '-').replace('?', '-')
        # nib.save(nifti_img,f'{OutputLocation}\\{roi_name}.nii')
        
    return

    # Load DICOM file with needle coordinates

    try:
        needle_path_file = glob.glob(os.path.join(InputDicomFolder, "RP*.dcm"))[0]
        ds = pydicom.dcmread(needle_path_file)
    except:
        print(f'Needle DICOM could not be read: {dicom_files[0].PatientName} - {InputDicomFolder.split('\\')[-1]}')
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
            print(points)
        
        break
        

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
            print(f'Needle Coordinate could not be read: {dicom_files[0].PatientName} - {InputDicomFolder.split('\\')[-1]} - Needle {i+1}')

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
    SeriesUIDFolders = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Filtered Data"
    OutputFolder = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Transformed Data"
    
    import time
    start_time = time.time()

    # i = 0
    for seriesFolder in os.listdir(SeriesUIDFolders):
        # i+=1
        # if i<333:
        #     continue
        
        # Create series folder if not already present
        if not os.path.exists(f'{OutputFolder}\\{seriesFolder}'):
            os.makedirs(f'{OutputFolder}\\{seriesFolder}') # Create folder with SeriesUID

        
        extractData(f'{SeriesUIDFolders}\\{seriesFolder}', f'{OutputFolder}\\{seriesFolder}')
        
        break
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    







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
    


