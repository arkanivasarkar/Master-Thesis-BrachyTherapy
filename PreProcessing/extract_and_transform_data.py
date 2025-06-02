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
   

    # Transpose array to [X, Y, Z] for saving as NifTI using nibabel
    pixel_matrix_ras = np.transpose(pixel_matrix, (2, 1, 0))  # [Z, Y, X] → [X, Y, Z]

    # Save as NIfTI
    nifti_image = nib.Nifti1Image(pixel_matrix_ras, affine_ras)
    nib.save(nifti_image, f'{OutputLocation}\\image.nii')
    

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
        roi_name = roi_name.replace('/', '-').replace('\\', '-').replace('?', '-')
        nib.save(nifti_img,f'{OutputLocation}\\{roi_name}.nii')
        
        

    # Load DICOM file with needle coordinates

    try:
        needle_path_file = glob.glob(os.path.join(InputDicomFolder, "RP*.dcm"))[0]
        ds = pydicom.dcmread(needle_path_file)
    except:
        print(f'Needle DICOM could not be read: {dicom_files[0].PatientName} - {InputDicomFolder.split('\\')[-1]}')
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
    
    for seriesFolder in os.listdir(SeriesUIDFolders):
        
        # Create series folder if not already present
        if not os.path.exists(f'{OutputFolder}\\{seriesFolder}'):
            os.makedirs(f'{OutputFolder}\\{seriesFolder}') # Create folder with SeriesUID

        
        extractData(f'{SeriesUIDFolders}\\{seriesFolder}', f'{OutputFolder}\\{seriesFolder}')

    







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
    


