# import nibabel as nib
# import pandas as pd

# # Load the NIfTI files
# nifti1 = nib.load(r"C:\Users\arkaniva\Downloads\OASIS_BrainParts_labels.DKT31.manual+aseg.MNI152.nii")
# nifti2 = nib.load(r"C:\Users\arkaniva\Documents\Segmentation-Segment_1-label.nii")

# # Get headers as dictionaries
# header1 = dict(nifti1.header)
# header2 = dict(nifti2.header)

# # Collect all unique keys from both headers
# all_keys = sorted(set(header1.keys()).union(header2.keys()))

# # Create a comparison table
# data = []
# for key in all_keys:
#     val1 = header1.get(key, 'N/A')
#     val2 = header2.get(key, 'N/A')
#     data.append([key, val1, val2, val1 == val2])

# # Create a DataFrame for display
# df = pd.DataFrame(data, columns=['Field', 'File 1', 'File 2', 'Match'])

# # Display or save the comparison table
# print(df.head())
# df.to_csv(r"C:\Users\arkaniva\Documents\nifti_comparison.csv", index=False)


# exit()

# import nibabel as nib

# # Load the NIfTI file
# img = nib.load(r"C:\Users\arkaniva\Downloads\pp.nii")  # or .nii

# # Get the header (contains metadata)
# header = img.header

# # Print header as a dictionary
# metadata = {key: header[key] for key in header.keys()}

# print(header)

# # Extract specific metadata
# print("Shape of data:", img.shape)
# print("Data type:", header.get_data_dtype())
# print("Voxel dimensions (pixdim):", header.get_zooms())
# print("Affine transformation matrix:\n", img.affine)
# print("Description:", header['descrip'].astype(str))

# exit()
# # Create a new header and update datatype to match zero_data
# new_header = header.copy()
# new_header.set_data_dtype(np.uint16)

# # Create a new NIfTI image
# new_img = nib.Nifti1Image(zero_data, affine, header=new_header)

# # Save the new NIfTI image
# nib.save(new_img, 'zero_image_with_metadata.nii.gz')

# exit()

# import numpy as np
# import nibabel as nib

# # Define shape (e.g., 128x128x64) and affine (usually identity or from a reference)
# shape = (128, 128, 64)
# affine = np.eye(4)

# # Create a labelmap with dummy labels (e.g., 0=background, 1=region A, 2=region B)
# labelmap_data = np.zeros(shape, dtype=np.uint16)

# # Example: Assign label 1 to a subregion
# labelmap_data[30:60, 30:60, 10:20] = 1
# labelmap_data[70:100, 70:100, 30:40] = 2

# # Create and save NIfTI labelmap
# labelmap_img = nib.Nifti1Image(labelmap_data, affine)
# nib.save(labelmap_img, r'C:\Users\arkaniva\Downloads\labelmap.nii')

# exit()
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from skimage.draw import polygon2mask
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label

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
    nib.save(nifti_image, f'{OutputLocation}\\pp.nii')
    

    # Load RT-STRUCT DICOM file to extract anatomy contours
    rt_file = glob.glob(os.path.join(InputDicomFolder, "RS*"))[0]
    rt_struct = pydicom.dcmread(rt_file)

    # Get ROI contours
    for i,roi in enumerate(rt_struct.StructureSetROISequence):
        roi_name = roi.ROIName
        contours =  rt_struct.ROIContourSequence[i].ContourSequence

        # Pre-allocate memory for binary mask
        mask_volume = np.zeros((pixel_matrix_ras.shape[2],pixel_matrix_ras.shape[0],pixel_matrix_ras.shape[1]), dtype=np.uint8)

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
        #     # z_dim, y_dim, x_dim = mask_volume.shape
        #     # interpolated_mask = np.zeros_like(mask_volume, dtype=np.uint8)

        #     # for y in range(y_dim):
        #     #     for x in range(x_dim):
        #     #         column = mask_volume[:, y, x]
        #     #         nonzero_indices = np.where(column > 0)[0]
        #     #         if len(nonzero_indices) == 0:
        #     #             continue  # nothing to interpolate

        #     #         # Interpolate in Z for each (y, x) pixel column
        #     #         interp_func = interp1d(
        #     #             nonzero_indices,
        #     #             column[nonzero_indices],
        #     #             kind="nearest",  # 'linear' works too, but 'nearest' preserves binary mask better
        #     #             bounds_error=False,
        #     #             fill_value=0
        #     #         )
        #     #         interpolated_mask[:, y, x] = interp_func(np.arange(z_dim))

        #     # for z in range(interpolated_mask.shape[0]):
        #     #     interpolated_mask[z] = binary_fill_holes(interpolated_mask[z])

        #     # labeled, _ = label(interpolated_mask)
        #     # largest_label = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
        #     # mask_volume = (labeled == largest_label).astype(np.uint8)



            for i in range(len(sorted_segment_slice_indexes)-1):
                slice_number_gap = sorted_segment_slice_indexes[i+1]-sorted_segment_slice_indexes[i]
                if slice_number_gap == 1:
                    continue
                for j in range(1, slice_number_gap):
                    alpha = i / (slice_number_gap + 1)
                    # interpolated = (1 - alpha) * mask_volume[sorted_segment_slice_indexes[i]] + alpha * mask_volume[sorted_segment_slice_indexes[i+1]]
                    # mask_volume[sorted_segment_slice_indexes[i]+j] = (interpolated > 0.5).astype(np.uint8)



                    sdf1 = signed_distance(mask_volume[sorted_segment_slice_indexes[i]])
                    sdf2 = signed_distance(mask_volume[sorted_segment_slice_indexes[i+1]])

                    # Interpolate
                    sdf_interp = (1 - alpha) * sdf1 + alpha * sdf2

                    mask_volume[sorted_segment_slice_indexes[i]+j]  = (sdf_interp > 0).astype(np.uint8)
                   

       
        mask_volume = np.transpose(mask_volume, (2, 1, 0))

        # Save masks as NIFTI files
        nifti_img = nib.Nifti1Image(mask_volume, affine_ras)
        nib.save(nifti_img,f'{OutputLocation}\\{roi_name}.nii')
        
        

    exit()

    # Load DICOM file with needle coordinates
    needle_path_file = glob.glob(os.path.join(InputDicomFolder, "RP*"))[0]
    ds = pydicom.dcmread(needle_path_file)
    num_needles = len(ds.ApplicationSetupSequence[0].ChannelSequence)
   
    # Extract all needle coordinates for each slice and convert to image coordinate system 
    needle_coordinates = {f'Needle {i}': [] for i in range(1,num_needles+1)}
   
    for i in range(num_needles):
        points = [controlPoints.ControlPoint3DPosition for controlPoints in ds.ApplicationSetupSequence[0].ChannelSequence[i].BrachyControlPointSequence]
        points = np.array(points)

        current_needle_coordinates = np.zeros(points.shape, dtype=np.int16)

        # Convert coordinates from patient coordinate system to image coordinate system
        for j in range(points.shape[0]):
            z_index = np.argmin(np.abs(np.array(z_positions) - points[j,2]))
            pixel_coords = np.round((points[j,:2] - origin[:2]) / spacing[:2]).astype(int)
            current_needle_coordinates[j,0], current_needle_coordinates[j,1], current_needle_coordinates[j,2] = map(int, np.append(pixel_coords,z_index))
                    
        needle_coordinates[f'Needle {i+1}'] = current_needle_coordinates.tolist()
    
    # Save Needle Coordinates in a JSON file
    with open(f'{OutputLocation}\\needle_coordinates.json','w') as file:
        json.dump(needle_coordinates, file, indent=4)
            


if __name__ == '__main__':
    InputDicomFolder = "C:\\Users\\arkaniva\\Downloads\\Testcase\\ef"
    OutputLocation  = "C:\\Users\\arkaniva\\Downloads"


    extractData(InputDicomFolder, OutputLocation)

    







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
    




# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load data
# data = pd.read_csv(r"C:\Users\arkaniva\Downloads\Phamton experiments (2).csv")
# df = data.tail(12)

# print(df)

# # Define color map
# color_map = {
#     'Air': '#87CEFA',    # light sky blue
#     '1X Buffer': '#D3D3D3',  # blue
#     'Oil': '#FFD700'     # oilish yellow
# }

# # Create plot
# plt.figure(figsize=(10, 6))
# ax = sns.barplot(
#     data=df,
#     x='AGAROSE DENSITY\n IN 100ml SOLUTION',
#     y='Percentage Change',
#     hue='STORING MEDIUM',
#     palette=color_map,
#     errorbar=None
# )

# # # Extract hue order used in the plot
# # hue_order = df['STORING MEDIUM'].unique()
# # x_order = df['AGAROSE DENSITY\n IN 100ml SOLUTION'].unique()
# # n_hue = len(hue_order)


# # # Annotate bars with hue labels
# # for i, p in enumerate(ax.patches):
# #     if i==3:
# #         continue
# #     x = p.get_x() + p.get_width() / 2
# #     y = p.get_height()
# #     hue_label = hue_order[i % n_hue]
# #     ax.annotate(hue_label, (x, y + 1), ha='center', va='bottom', fontsize=10)


# # Set the labels manually
# ax.set_xticklabels(['0.5%', '1%', '5%', '10%'])
# plt.title('Change in Phantom Weight with 1X Buffer as Solvent')
# plt.xlabel('Agarose Density')
# plt.ylabel('Absolute Percentage Reduction in Weight')
# plt.legend(title='Storage Media')
# # plt.tight_layout()

# # Show plot
# plt.show()
