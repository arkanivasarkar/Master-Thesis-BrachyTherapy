

from PIL import Image
import pytesseract
import pandas as pd
import cv2
# If you're on Windows and Tesseract is not in PATH, set the full path to the executable:
# Replace this path with where Tesseract is installed on your system
# Usually: C:\Program Files\Tesseract-OCR\tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\sarkaraa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

import cv2
import pytesseract
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Load the image
image_path = 'image_2.png'
image = cv2.imread(r'W:\strahlenklinik\science\Physik\Arkaniva\Prostataexport Arkaniva Sarkar\Codes\Master-Thesis-BrachyTherapy\PreProcessing\image_2.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binnedArray = (gray // 5) * 5

binnedArray = (binnedArray>170) & (binnedArray<255)

# plt.imshow(binnedArray,cmap='gray') 
# plt.show()
# exit()
# Invert image because text is white on black


# Run OCR
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(binnedArray, config=custom_config)

print("OCR Output:\n")
print(text)

# Perform OCR with bounding box info
data = pytesseract.image_to_data(binnedArray, output_type=pytesseract.Output.DATAFRAME)

# Filter out empty text entries
data = data[data['text'].notnull() & (data['text'].str.strip() != '')]

# Show results
for index, row in data.iterrows():
    print(f"Text: '{row['text']}' at ({row['left']}, {row['top']}, {row['width']}, {row['height']})")
