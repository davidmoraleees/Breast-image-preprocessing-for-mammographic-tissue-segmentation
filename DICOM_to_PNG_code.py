# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:59:26 2024

@author: david
"""

import os
import pydicom
import imageio
from tqdm import tqdm

# Directorio donde se encuentran las imágenes DICOM
input_dir = "C:/Users/david/Downloads/AllDICOMs"

# Directorio donde se guardarán las imágenes PNG
output_dir = "C:/Users/david/Downloads/AllDICOMs_PNG"

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtener la lista de archivos DICOM en el directorio de entrada
dicom_files = [filename for filename in os.listdir(input_dir) if filename.endswith(".dcm")]

# Configurar la barra de progreso
progress_bar = tqdm(total=len(dicom_files), desc="Conversion progreso")

# Iterar sobre todos los archivos DICOM
for filename in dicom_files:
    # Ruta completa al archivo DICOM
    dicom_file = os.path.join(input_dir, filename)
    
    # Cargar el archivo DICOM
    dicom_data = pydicom.dcmread(dicom_file)
    
    # Obtener los píxeles de la imagen
    image = dicom_data.pixel_array
    
    # Nombre de archivo de salida (reemplazando la extensión .dcm con .png)
    output_filename = os.path.splitext(filename)[0] + ".png"
    
    # Ruta completa al archivo de salida
    output_file = os.path.join(output_dir, output_filename)
    
    # Guardar la imagen como PNG
    imageio.imwrite(output_file, image)
    
    # Actualizar la barra de progreso
    progress_bar.update(1)

# Cerrar la barra de progreso
progress_bar.close()

print("¡Conversión completa!")







