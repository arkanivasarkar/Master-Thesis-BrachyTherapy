import os
import glob
import shutil
from datetime import datetime
import pydicom


path = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Raw Data"
outputFolder = r"W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Filtered Data"

for folderGroup in os.listdir(path):
    for folder in os.listdir(f'{path}\\{folderGroup}'):

        # fetch all ultrasound
        ultrasound_files = glob.glob(f'{path}\\{folderGroup}\\{folder}\\US*')
        if len(ultrasound_files)==0:
            continue

        #TODO:
        # create folder with serie
        # compare needles and structure folder date
        # save series
        for files in ultrasound_files:
            ds = pydicom.dcmread(files, stop_before_pixels=True)
            series_uid = ds.SeriesInstanceUID          

            # make patient folder
            if not os.path.exists(f'{outputFolder}\\{series_uid}'):
                os.makedirs(f'{outputFolder}\\{series_uid}')

                series_date = datetime.strptime(ds.StudyDate, '%Y%m%d').date()

                for item in os.listdir(f'{path}\\{folderGroup}\\{folder}'):
                    if not os.path.isdir(f'{path}\\{folderGroup}\\{folder}\\{item}'):
                        continue

                    folderDate = datetime.strptime(item.split()[0], '%Y-%m-%d').date()
                    
                    if series_date != folderDate:
                        continue
                    
                    datafiles = glob.glob(f'{path}\\{folderGroup}\\{folder}\\{item}\\*')

                    for i in datafiles:
                        shutil.copy2(i, f'{outputFolder}\\{series_uid}\\{i.split('\\')[-1]}')

            shutil.copy2(files, f'{outputFolder}\\{series_uid}\\{files.split('\\')[-1]}')

       
            # date2_str = '2025-05-07 hPr_2025-05-07'
            # date2 = datetime.strptime(date2_str.split()[0], '%Y-%m-%d').date()

           


