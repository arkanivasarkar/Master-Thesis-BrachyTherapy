import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from skimage.draw import polygon2mask
from scipy.ndimage import distance_transform_edt

def signed_distance(mask):
    mask = (mask > 0).astype(np.uint8)
    dist_out = distance_transform_edt(mask == 0)
    dist_in = distance_transform_edt(mask == 1)
    return dist_in - dist_out


def extractData(InputDicomFolder, OutputLocation):

    # Load US image volume from DICOM folder and extract relevant metadata
    image_slices_filenames = glob.glob(os.path.join(InputDicomFolder, "US*"))

    # Read all DICOM files in the directory
    files = [(pydicom.dcmread(filename),filename) for filename in image_slices_filenames]

    # Sort the DICOM files by ImagePositionPatient to ensure correct slice order
    files.sort(key=lambda x: int(x[0].ImagePositionPatient[2]))

    dicom_files = [item[0] for item in files]
    sorted_filenames = [item[1] for item in files]
   
    # # Get slice z-positions
    z_positions = [float(ds.ImagePositionPatient[2]) for ds in dicom_files]


    # Orientation vectors
    x_cosine = np.array(dicom_files[0].ImageOrientationPatient[0:3])
    y_cosine = np.array(dicom_files[0].ImageOrientationPatient[3:6])
    z_cosine = np.cross(x_cosine, y_cosine)


    # Origin
    origin = [float(val) for val in dicom_files[0].ImagePositionPatient]
    

    # Affine matrix
    affine = np.eye(4)
    affine[:3, 0] = x_cosine * dicom_files[0].PixelSpacing[0]
    affine[:3, 1] = y_cosine * dicom_files[0].PixelSpacing[1]
    affine[:3, 2] = z_cosine * dicom_files[0].SliceThickness
    affine[:3, 3] = dicom_files[0].ImagePositionPatient


    # Convert LPS → RAS for NIfTI compliance
    lps_to_ras = np.diag([-1, -1, 1, 1])
    affine_ras = lps_to_ras @ affine

    # Extract pixel matrix
    pixel_arrays = [ds.pixel_array for ds in dicom_files] 
    pixel_matrix = np.stack(pixel_arrays, axis=0) # [Z,Y,X] or [slice, column, row]
    
    img = (pixel_matrix // 5) * 5
    
    mask = img == 255
    img[mask] = 0

    # plt.imshow(img[6,:,:], cmap='gray')
    # plt.show()
    
    from skimage.util.shape import view_as_windows
 
    plus_kernel = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)
    
    # print(img.shape)

    # Get indices of non-zero elements in the kernel
    plus_indices = np.argwhere(plus_kernel)

    # Extract 3x3 sliding windows over the image
    windows = view_as_windows(img[6,:,:], (3, 3))

    # Initialize mask for detected plus centers
    result_mask = np.zeros_like(img[6,:,:], dtype=bool)
    plus_centers = []
    
    
    # Loop over valid positions in the image
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            patch = windows[i, j]
            values = patch[plus_kernel == 1]
            if np.all(values == values[0]) and 200 <= values[0] <= 255:
                center_y, center_x = i + 1, j + 1  # center of 3x3 patch
                plus_centers.append((center_x, center_y))  # x first for plotting
                
                
                # Mark the 5 corresponding positions as True
                for dy, dx in plus_indices:
                    result_mask[i + dy, j + dx] = True
        
    plus_centers = np.array(plus_centers)
    
    tolerance = 3
    spacing = int(5//dicom_files[0].PixelSpacing[0])
    
    from collections import defaultdict

    def filter_aligned_points(points):
        x_groups = defaultdict(list)  # points with the same x
        y_groups = defaultdict(list)  # points with the same y

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

        return np.array(list(result))
    
    
    plus_centers = filter_aligned_points(plus_centers)
   
    def zero_out_neighbors(grid, x, y):
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0,0)]  # up, down, left, right

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                grid[nx][ny] = 0
        return grid
    
    def apply_3x3_average(image, x, y):
        rows, cols = image.shape
        kernel_values = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    kernel_values.append(image[nx, ny])

        # Compute average and assign to center pixel
        if kernel_values:
            image[x, y] = np.mean(kernel_values)
            
        return image
                
    for pt in plus_centers:
        img[6,:,:] = zero_out_neighbors(img[6,:,:], pt[1], pt[0])
        
        directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),  (0,0),  ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
        ]  


        for _ in range(3):
            for dx, dy in directions:
                nx, ny = pt[1] + dx, pt[0] + dy
                
                img[6,:,:] = apply_3x3_average(img[6,:,:], nx,ny)
            
       
       
        plt.imsave( rf"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Test\{ InputDicomFolder.split('\\')[-1]}.png", img[6,:,:] , cmap='gray')   

    #print(plus_centers)
    
    # coords=[]
    # for centers in plus_centers:
        
    #     plt.scatter(centers[0], centers[1], color='red', s=10)
    #     plt.imshow(result_mask, cmap='gray')
    #     plt.show()
        
        
    #     x_diff = np.abs(plus_centers[:, 0] - centers[0])
    #     y_diff = np.abs(plus_centers[:, 1] - centers[1])

    #     # Check if any x or y difference is between 41 and 43
    #     mask = ((x_diff >= spacing-1) & (x_diff <= spacing+1) & y_diff==0) & (x_diff==0 & (y_diff >= spacing-1) & (y_diff <= spacing+1))

    #     # distances = np.round(np.linalg.norm(plus_centers - centers, axis=1))
    #     # #print(distances)
    #     # mask = (distances >= spacing-1) & (distances <= spacing+1)
        
    #     #print(plus_centers.shape)
    #     if 2<=mask.sum()<=4:
    #         print(plus_centers[mask])
    #         coords.append(centers)
            
    #         pt=plus_centers[mask]
            
    #         print( np.round(np.linalg.norm(pt - centers, axis=1)))
    #         plt.scatter(centers[0], centers[1], color='red', s=10)
    #         plt.scatter(pt[:,0], pt[:,1], color='green', s=10)
    #         plt.imshow(result_mask, cmap='gray')
    #         plt.show()
    #         break
        
        
    # plus_centers = np.array(coords)

        
        
        
        
        

    # plt.scatter(plus_centers[:, 0], plus_centers[:, 1], color='red', s=10)
    # #
    # # 
    # # plt.imshow(img[6,:,:], cmap='gray')
    # plt.show()
    
    
        
    return
   

    # Transpose array to [X, Y, Z] for saving as NifTI using nibabel
    pixel_matrix_ras = np.transpose(pixel_matrix, (2, 1, 0))  # [Z, Y, X] → [X, Y, Z]

    # Save as NIfTI
    #nifti_image = nib.Nifti1Image(pixel_matrix_ras, affine_ras)
    #nib.save(nifti_image, f'{OutputLocation}\\image.nii')
    

    
    
    return

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
    
    # i = 0
    for seriesFolder in os.listdir(SeriesUIDFolders):
        # i+=1
        # if i<333:
        #     continue
        
        # Create series folder if not already present
        if not os.path.exists(f'{OutputFolder}\\{seriesFolder}'):
            os.makedirs(f'{OutputFolder}\\{seriesFolder}') # Create folder with SeriesUID

        
        extractData(f'{SeriesUIDFolders}\\{seriesFolder}', f'{OutputFolder}\\{seriesFolder}')
        
        # break

    







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
    


