import os
import glob
import json
import numpy as np
import nibabel as nib
import pydicom
from scipy.ndimage import distance_transform_edt, convolve, binary_dilation
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import filters
from collections import Counter

def signed_distance(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    dist_in = distance_transform_edt(binary)
    dist_out = distance_transform_edt(~binary)
    return dist_in - dist_out


def filter_collinear(points: np.ndarray, min_span: int = 8) -> np.ndarray:
    # vectorized grouping via counts
    xs, ys = points[:, 0], points[:, 1]
    x_counts = Counter(xs)
    y_counts = Counter(ys)
    mask = np.array([x_counts[x] > min_span or y_counts[y] > min_span for x, y in points])
    return points[mask]


def detect_plus_mask(binned: np.ndarray, lo=200, hi=255) -> np.ndarray:
    # cross-shaped kernel convolution to detect plus markers
    kernel = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=int)
    matches = convolve(((binned >= lo) & (binned <= hi)).astype(int), kernel, mode='constant')
    # center plus neighbors => sum must be 5
    centers = (matches == 5)
    # remove border
    centers[[0,-1], :] = False
    centers[:, [0,-1]] = False
    return centers


def fast_fill(slice_img: np.ndarray, kernel: np.ndarray, iterations: int = 3) -> None:
    bad = slice_img < 0
    if not bad.any():
        return
    for _ in range(iterations):
        bad = slice_img < 0
        if not bad.any(): break
        sums = convolve(np.where(~bad, slice_img, 0.0), kernel, mode='constant')
        counts = convolve((~bad).astype(float), kernel, mode='constant')
        update = bad & (counts > 0)
        slice_img[update] = (sums[update] / counts[update])


def extractData(input_folder: str, output_path: str) -> None:
    # load DICOM series
    paths = sorted(glob.glob(os.path.join(input_folder, 'US*')),
                   key=lambda p: float(pydicom.dcmread(p).ImagePositionPatient[2]))
    dicoms = [pydicom.dcmread(p) for p in paths]
    # build affine
    first = dicoms[0]
    x_cos, y_cos = np.array(first.ImageOrientationPatient[:3]), np.array(first.ImageOrientationPatient[3:6])
    z_cos = np.cross(x_cos, y_cos)
    affine = np.eye(4)
    affine[:3,0] = x_cos * first.PixelSpacing[0]
    affine[:3,1] = y_cos * first.PixelSpacing[1]
    affine[:3,2] = z_cos * first.SliceThickness
    affine[:3,3] = first.ImagePositionPatient
    affine_ras = np.diag([-1,-1,1,1]) @ affine

    # stack pixel data
    volume = np.stack([ds.pixel_array for ds in dicoms], axis=0).astype(np.float32)
    orig = volume.copy()

    # bin intensities
    binned = (volume.astype(int) // 5) * 5
    # mark contours
    volume[binned == 255] = -1

    # detection and marking
    kernel_cross = np.ones((3,3), dtype=np.uint8)
    for z in range(volume.shape[0]):
        centers = detect_plus_mask(binned[z])
        if not centers.any(): continue
        coords = np.column_stack(np.where(centers))
        # filter linearly
        pts = np.stack([coords[:,1], coords[:,0]], axis=1)
        pts = filter_collinear(pts)
        if pts.size == 0: continue
        # dilate and mark
        mask = np.zeros_like(centers)
        mask[coords[:,0], coords[:,1]] = True
        mask = binary_dilation(mask, structure=np.ones((3,3)))
        volume[z][mask] = -1

    # fill holes
    avg_kernel = np.ones((3,3), dtype=np.float32)
    for z in range(volume.shape[0]):
        fast_fill(volume[z], avg_kernel)

    # quality metrics
    psnr = peak_signal_noise_ratio(orig.astype(np.uint8), volume.astype(np.uint8))
    ssim = structural_similarity(orig.astype(np.uint8), volume.astype(np.uint8))
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    # save NIFTI
    ras = np.transpose(volume, (2,1,0)).astype(np.uint8)
    img = nib.Nifti1Image(ras.astype(np.uint8), affine_ras)
    nib.save(img, 'image2.nii')
    # nib.save(img, os.path.join(OutputLocation, "image.nii"))

    return

    
    
    with open(r'C:\Users\sarkaraa\Downloads\roi_mapping.json', "r") as f:
        roi_dict = json.load(f)

    # Example: print the dictionary or use it
    

    # Load RT-STRUCT DICOM file to extract anatomy contours
    try:
        rt_file = glob.glob(os.path.join(InputDicomFolder, "RS*"))[0]
        rt_struct = pydicom.dcmread(rt_file)
    except:
        # print(f'RT Struct not found: {dicom_files[0].PatientName} - {InputDicomFolder.split('\\')[-1]}')
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
    


