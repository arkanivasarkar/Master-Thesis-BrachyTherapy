import os
import glob
import numpy as np
import pydicom
import pandas as pd


def extractData(InputDicomFolder):

    # Load US image volume from DICOM folder and extract relevant metadata
    image_slices_filenames = glob.glob(os.path.join(InputDicomFolder, "US*"))

    # Read all DICOM files in the directory
    files = [(pydicom.dcmread(filename),filename) for filename in image_slices_filenames]
 

    # Sort the DICOM files by ImagePositionPatient to ensure correct slice order
    files.sort(key=lambda x: int(x[0].ImagePositionPatient[2]))

    dicom_files = [item[0] for item in files]

    pixel_arrays = [ds.pixel_array for ds in dicom_files] 
    pixel_matrix = np.stack(pixel_arrays, axis=0)

    # Orientation vectors
    x_cosine = np.array(dicom_files[0].ImageOrientationPatient[0:3])
    y_cosine = np.array(dicom_files[0].ImageOrientationPatient[3:6])
    z_cosine = np.cross(x_cosine, y_cosine)

    patient_name = dicom_files[0].PatientName
    seriesUID = InputDicomFolder.split('\\')[-1]
    rows = pixel_matrix.shape[1]
    columns = pixel_matrix.shape[2]
    nframes = pixel_matrix.shape[0]
    pixel_size_row = dicom_files[0].PixelSpacing[0]
    pixel_size_col = dicom_files[0].PixelSpacing[1]
    slice_thickness = dicom_files[0].SliceThickness
    min_intensity = np.min(pixel_matrix)
    max_intensity = np.max(pixel_matrix)
    volume_normal = z_cosine[2]
   

   
    # Load RT-STRUCT DICOM file to extract anatomy contours
    try:
        rt_file = glob.glob(os.path.join(InputDicomFolder, "RS*"))[0]
        rt_struct = pydicom.dcmread(rt_file)
        rt_struct_present = True

        # Get ROI contours        
        num_roi = 0
        for i,roi in enumerate(rt_struct.StructureSetROISequence):
            if i == 0:
                roi_names = roi.ROIName
            else:
                roi_names = ",".join([roi_names, roi.ROIName]) 
            num_roi +=1

    except:
        rt_struct_present = False
        num_roi = 0
        roi_names = ''
 

    # Load DICOM file with needle coordinates
    try:
        needle_path_file = glob.glob(os.path.join(InputDicomFolder, "RP*.dcm"))[0]
        ds = pydicom.dcmread(needle_path_file)
        needle_file_present = True
        num_needles = 0
        missing_needles = 0
        for i in range( len(ds.ApplicationSetupSequence[0].ChannelSequence)):
            try:                
                points = [controlPoints.ControlPoint3DPosition for controlPoints in ds.ApplicationSetupSequence[0].ChannelSequence[i].BrachyControlPointSequence]
                num_needles +=1
            except:
                missing_needles+=1
    except:
        needle_file_present = False
        num_needles = 0
        missing_needles = None

    
    return {'PatientName': patient_name,
            'SeriesUID': seriesUID,
            'Rows': rows,
            'Columns': columns,
            'Frames': nframes,
            'Pixel_Size_Row': pixel_size_row,
            'Pixel_Size_Col': pixel_size_col,
            'Slice_Thickness': slice_thickness,
            'Imin': min_intensity,
            'Imax': max_intensity,
            'volume_normal': volume_normal,
            'ispresent_RTStruct': rt_struct_present,
            'Num_ROI': num_roi,
            'ROI_Names': roi_names,
            'ispresent_NeedleFile': needle_file_present,
            'Num_Needles': num_needles,
            'Missing_Needles': missing_needles
            }
    
   


if __name__ == '__main__':
    SeriesUIDFolders = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Filtered Data"
    OutputFolder = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar"
    
    descriptions = []
    for seriesFolder in os.listdir(SeriesUIDFolders):
        metadata = extractData(f'{SeriesUIDFolders}\\{seriesFolder}')
        descriptions.append(metadata)
        

    df = pd.DataFrame(descriptions)
    df.to_csv(f'{OutputFolder}\\DataDescription.csv', index=False)






