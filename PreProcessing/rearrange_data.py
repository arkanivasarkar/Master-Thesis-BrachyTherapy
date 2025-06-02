'''
Code to read all folders exported from Oncentra and save it in different folders based on SeriesUID
'''

import os
import glob
import shutil
from datetime import datetime
import pydicom


inputPath = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Raw Data"
outputFolder = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Filtered Data"

for folderGroup in os.listdir(inputPath): #For all Oncentra folders
    for folder in os.listdir(f'{inputPath}\\{folderGroup}'):   #For all patient folders

        # Fetch all ultrasound dicom images (files start with prefix 'US')
        ultrasound_files = glob.glob(f'{inputPath}\\{folderGroup}\\{folder}\\US*')

        # If no ultrasound dicom files detected, skip loop (first filter)
        if len(ultrasound_files)==0:
            continue

        
        # Read all ultrasound files in a folder
        for files in ultrasound_files:
            ds = pydicom.dcmread(files, stop_before_pixels=True)
            series_uid = ds.SeriesInstanceUID          

            # Create series folder if not already present
            if not os.path.exists(f'{outputFolder}\\{series_uid}'):
                os.makedirs(f'{outputFolder}\\{series_uid}') # Create folder with SeriesUID

                # Find needle coordinates, RTSTRUCT and dose dicoms 
                # and move to series folder by comparing series dates and folder date
                series_date = datetime.strptime(ds.StudyDate, '%Y%m%d').date()

                for item in os.listdir(f'{inputPath}\\{folderGroup}\\{folder}'):
                    if not os.path.isdir(f'{inputPath}\\{folderGroup}\\{folder}\\{item}'):
                        continue

                    folderDate = datetime.strptime(item.split()[0], '%Y-%m-%d').date()
                    
                    if series_date != folderDate:
                        continue
                    
                    datafiles = glob.glob(f'{inputPath}\\{folderGroup}\\{folder}\\{item}\\*')

                    # Move the files to series folder
                    for i in datafiles:
                        shutil.copy2(i, f'{outputFolder}\\{series_uid}\\{i.split('\\')[-1]}')


            # Move the ultrasound DICOM to SeriesUID folder
            shutil.copy2(files, f'{outputFolder}\\{series_uid}\\{files.split('\\')[-1]}')

       
           


