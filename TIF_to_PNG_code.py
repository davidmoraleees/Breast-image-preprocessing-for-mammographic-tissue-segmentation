import os
import imageio
from tqdm import tqdm

# Directorio donde se encuentran las imágenes TIFF
input_dir = r"C:\Users\david\OneDrive\Escriptori\DAVID\4_Física UB\8è semestre primavera 2024\Processament d'imatge i visió artificial (PIVA)\Breast cancer diagnosis\ROI_Masks"

# Directorio donde se guardarán las imágenes PNG
output_dir = r"C:\Users\david\OneDrive\Escriptori\DAVID\4_Física UB\8è semestre primavera 2024\Processament d'imatge i visió artificial (PIVA)\Breast cancer diagnosis\ROI_Masks_PNG"

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtener la lista de archivos TIFF en el directorio de entrada
tif_files = [filename for filename in os.listdir(input_dir) if filename.endswith(".tif") or filename.endswith(".tiff")]

# Configurar la barra de progreso
progress_bar = tqdm(total=len(tif_files), desc="Conversion progreso")

# Iterar sobre todos los archivos TIFF
for filename in tif_files:
    # Ruta completa al archivo TIFF
    tif_file = os.path.join(input_dir, filename)
    
    # Cargar la imagen TIFF
    image = imageio.imread(tif_file)
    
    # Nombre de archivo de salida (reemplazando la extensión .tif/.tiff con .png)
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
