# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:59:26 2024

@author: david
"""

import os
import pydicom
import imageio
from tqdm import tqdm
import numpy as np

input_dir = "/Users/anastasiask/Documents/Física/UB/PIVA/Final_Project/AllDICOMs" #INBreast dataset
output_dir = "/Users/anastasiask/Documents/Física/UB/PIVA/Final_Project/AllDICOMs_PNG" 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dicom_files = [filename for filename in os.listdir(input_dir) if filename.endswith(".dcm")]
progress_bar = tqdm(total=len(dicom_files), desc="Conversion in progress")

for filename in dicom_files:
    dicom_file = os.path.join(input_dir, filename)
    dicom_data = pydicom.dcmread(dicom_file)
    image = dicom_data.pixel_array

    if np.max(image) > 1:
        image = image / np.max(image)

    image_uint8 = (image * 255).astype(np.uint8)
    output_filename = os.path.splitext(filename)[0] + ".png"
    output_file = os.path.join(output_dir, output_filename)
    imageio.imwrite(output_file, image_uint8)
    progress_bar.update(1)

progress_bar.close()

print("Conversion completed")
