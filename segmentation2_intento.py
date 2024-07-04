import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments, moments_hu
from skimage.util import img_as_ubyte
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.colors as mcolors

# Cargar la imagen
image_path = "INbreast/AllDICOMs_PNG/20586908_6c613a14b80a8591_MG_R_CC_ANON.png"
image = io.imread(image_path)

# Convertir la imagen a escala de grises si es necesario
if len(image.shape) == 3:
    gray_image = color.rgb2gray(image)
else:
    gray_image = image

# Mostrar la imagen original
plt.imshow(gray_image, cmap='gray')
plt.title('Imagen Original')
plt.show()

# Aplanar la imagen a 2D para aplicar KMeans
flat_image = gray_image.reshape(-1, 1)

# Aplicar KMeans para segmentar la imagen en 5 clases (4 tejidos + fondo)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(flat_image)
segmented_labels = kmeans.labels_

# Reconstruir la imagen segmentada
segmented_image = segmented_labels.reshape(gray_image.shape)

# Mostrar la imagen segmentada
plt.imshow(segmented_image, cmap='gray')
plt.title('Imagen Segmentada con KMeans')
plt.show()

# Función para extraer características de cada segmento
def extract_features(segmented_image, gray_image):
    features = []
    for label in np.unique(segmented_image):
        mask = segmented_image == label
        
        # Asegurar que el parche tenga estructura 2D correcta
        if np.sum(mask) == 0:
            print(f"Skipping segment {label} due to insufficient size: {np.sum(mask)} pixels")
            continue
        
        patch = gray_image * mask
        
        # Verificar si el parche no está vacío y tiene tamaño suficiente
        if np.sum(mask) < 10:  # reducir el tamaño mínimo
            print(f"Skipping segment {label} due to insufficient size: {np.sum(mask)} pixels")
            continue
        
        mean_intensity = np.mean(patch[mask])
        std_intensity = np.std(patch[mask])
        
        # Convertir parche a uint8 y verificar dimensiones
        patch = img_as_ubyte(patch)
        if patch.ndim != 2 or patch.shape[0] == 0 or patch.shape[1] == 0:  # Ajuste para garantizar la dimensionalidad correcta
            print(f"Skipping segment {label} due to invalid patch dimensions: {patch.shape}")
            continue
        
        # Calcular GLCM solo si el parche tiene suficiente tamaño
        if np.sum(mask) >= 50:
            glcm = graycomatrix(patch, [1], [0], 256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            ASM = graycoprops(glcm, 'ASM')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
        else:
            contrast = dissimilarity = homogeneity = ASM = energy = correlation = 0
        
        # Calcular los momentos
        moments_values = moments(patch.astype(np.float64))
        hu_moments = moments_hu(moments_values)
        
        feature_vector = np.hstack([mean_intensity, std_intensity, contrast, dissimilarity, homogeneity, ASM, energy, correlation, hu_moments])
        features.append(feature_vector)
    
    return np.array(features)

# Extraer características de los segmentos
features = extract_features(segmented_image, gray_image)

# Verificar si se han extraído características
if len(features) == 0:
    raise ValueError("No se han extraído características de la imagen. Verifica el proceso de segmentación.")

# Verificar y reemplazar valores infinitos y extremadamente grandes
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

print("Características extraídas:", features.shape)

# Preparar las etiquetas (ejemplo: BI-RADS para cada segmento)
labels = np.array([2] * len(features))  # Sustituye [2] por las etiquetas reales

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Entrenar un clasificador Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Realizar predicciones
y_pred = clf.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_pred))

# Especificar colores para cada clase usando nombres
color_names = ["black", "green", "yellow", "blue", "red"]
colors = [mcolors.to_rgb(name) for name in color_names]

# Crear una imagen coloreada a partir de las etiquetas predichas
segmented_image_colored = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.float32)

for i, color in enumerate(colors):
    segmented_image_colored[segmented_image == i] = color

# Convertir la imagen coloreada a formato uint8
segmented_image_colored = (segmented_image_colored * 255).astype(np.uint8)

# Mostrar la imagen segmentada con colores
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Imagen original
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Imagen Original')

# Imagen segmentada con colores diferentes para cada clase
ax[1].imshow(segmented_image_colored)
ax[1].set_title('Imagen Segmentada')
plt.savefig('segmentada_kmeans_colores.png')
plt.show()
