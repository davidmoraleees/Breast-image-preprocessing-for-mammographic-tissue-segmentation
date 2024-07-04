import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import os

# Función para leer el archivo bi-rads.txt
def read_birads_file(filepath):
    birads_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_id, birads = parts
                birads_dict[image_id] = int(birads)
    return birads_dict

# Leer el archivo bi-rads.txt
birads_file_path = "bi-rads.txt"
birads_dict = read_birads_file(birads_file_path)

# Función para extraer la ID de la imagen a partir del nombre del archivo
def extract_image_id(filename):
    base_name = os.path.basename(filename)
    image_id = base_name.split('_')[0]
    return image_id

# Path de la imagen
image_path = "INbreast/AllDICOMs_PNG/20586908_6c613a14b80a8591_MG_R_CC_ANON.png"

# Extraer la ID de la imagen
current_image_id = extract_image_id(image_path)

# Obtener la etiqueta BI-RADS para la imagen actual
birads_label = birads_dict.get(current_image_id, None)

if birads_label is None:
    raise ValueError(f"No BI-RADS label found for image ID {current_image_id}")

# Cargar la imagen
image = io.imread(image_path)

# Convertir la imagen a escala de grises si es necesario
if len(image.shape) == 3:
    gray_image = color.rgb2gray(image)
else:
    gray_image = image

# Aplanar la imagen a 2D para aplicar KMeans
flat_image = gray_image.reshape(-1, 1)

# Aplicar KMeans para segmentar la imagen en 5 clases (incluyendo el fondo)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(flat_image)
segmented_labels = kmeans.labels_

# Reconstruir la imagen segmentada
segmented_image = segmented_labels.reshape(gray_image.shape)

# Especificar colores para cada clase usando nombres
color_names = ["red", "black", "gray", "yellow", "blue"]
colors = [mcolors.to_rgb(name) for name in color_names]

# Crear una imagen coloreada a partir de las etiquetas de KMeans
segmented_image_colored = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.float32)

for i in range(5):
    segmented_image_colored[segmented_image == i] = colors[i]

# Convertir la imagen coloreada a formato uint8
segmented_image_colored = (segmented_image_colored * 255).astype(np.uint8)

# Visualizar la segmentación con diferentes colores para cada clase
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Imagen original
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Imagen Original')

# Imagen segmentada con colores diferentes para cada clase
ax[1].imshow(segmented_image_colored)
ax[1].set_title('Imagen Segmentada')
plt.show()
