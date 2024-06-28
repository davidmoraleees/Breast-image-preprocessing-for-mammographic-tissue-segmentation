# Breast-cancer-diagnosis
Escribo los pasos que tenemos que seguir. Yo dejaría la selección automática de imágenes para el final, y empezaría primero por el pre-procesamiento de imágenes (punto 2).

También, comentar que inicialmente estaba usando el dataset de los links que pone figshare, pero he visto que no podemos continuar con ese dataset. Así que de momento sigo con el dataset de INbreast, te adjunto también el link. Descarga el archivo, descomprímelo y ponlo en la carpeta donde tengas el proyecto, y llama a esta carpeta 'INbreast'. Luego tienes que crear una carpeta que se llame 'AllDICOMs_PNG' dentro de la carpeta 'INbreast', y tienes que ejecutar el código para pasar de formato DICOM a formato PNG, cambiando los paths correspondientes.

El código principal donde pondremos todo es el main.py.

## 1. Selección Automática de Imágenes

### Extracción de Parámetros
1. Extraer parámetros de los encabezados DICOM de las imágenes mamográficas, incluyendo:
   - Edad del paciente
   - Dosis del órgano
   - Dosis de entrada
   - Exposición (mAs)
   - Exposición relativa de rayos X
   - Fuerza de compresión
   - Grosor de la parte del cuerpo
   - kVp (kilovoltaje pico)

### Atributos Basados en Imágenes
2. Calcular atributos basados en las imágenes:
   - Porcentaje de área periférica (PPA)
   - Porcentaje de cobertura pectoral (PPC)
   - Porcentaje de cobertura de la línea de piel (PSC)

### Segmentación Automática
3. Usar el algoritmo de Otsu para segmentar automáticamente las áreas periféricas del seno.

### Modelo de Probabilidad
4. Utilizar técnicas de aprendizaje automático (por ejemplo, Random Forest) para construir un modelo de probabilidad que determine qué imágenes requieren pre-procesamiento.

## 2. Pre-Procesamiento de Imágenes

### 2.1. Separación de la Periferia del Seno
1. **Umbralización de Otsu**: Aplicar el umbral de Otsu para segmentar la región del seno.
2. **Mejora de Segmentación**: Aplicar un umbral adicional basado en el valor medio de los píxeles del área periférica segmentada.
3. **Relleno Morfológico**: Rellenar agujeros pequeños y conectar píxeles vecinos usando un elemento estructurante de 3x3.
4. **Refinamiento de BPA**: Mantener solo el componente más grande conectado para refinar el área periférica del seno (BPA).

### 2.2. Propagación de la Relación de Intensidad
1. **Mapa de Distancia**: Generar un mapa de distancia calculando la distancia más corta de cada píxel a la línea de piel.
2. **Corrección de Píxeles**: Multiplicar el valor de gris de cada píxel por una relación de propagación de intensidad calculada en una vecindad de 17x17 píxeles.

### 2.3. Estimación del Grosor del Seno
1. **Par de Vistas CC y MLO**: Usar un par de vistas CC y MLO para aproximar la forma del seno y estimar la relación de grosor.
2. **Extracción de Línea de Piel**: Extraer y dividir la línea de piel en la vista MLO en líneas superior e inferior.
3. **Cálculo de Relación de Grosor**: Calcular la relación de grosor basado en la longitud de líneas paralelas generadas desde la línea de piel hasta la pared torácica.

### 2.4. Balanceo de Intensidad
1. **Normalización Logarítmica**: Normalizar logarítmicamente las relaciones de grosor estimadas.
2. **Corrección de Valores de Píxeles**: Ajustar el valor de gris de cada píxel en el área periférica basado en la proporción relativa de grosor del seno, utilizando la relación de grosor global como referencia.

## 3. Segmentación y Clasificación Mamográfica

### Segmentación de Imágenes
1. Segmentar las imágenes procesadas en tejidos nodulares, lineales, homogéneos y radiolúcidos.

### Extracción de Características
2. Extraer características de textura y geométricas usando histogramas de niveles de gris y momentos geométricos.

### Selección y Clasificación de Características
3. Usar algoritmos de selección de atributos y clasificadores (e.g., Random Committee) para seleccionar las características más discriminativas y clasificar los píxeles en las categorías de riesgo de Tabár y BI-RADS.

### Uso de los Informes Médicos
1. Integrar la información de los informes médicos para validar y ajustar las clasificaciones automáticas. Los informes proporcionan:
   - Composición del seno (e.g., predominantemente fibrograso)
   - Clasificación BI-RADS (e.g., BIRADS 1, 3, 5)
   - Hallazgos específicos (e.g., opacidades, calcificaciones, adenopatías)

## 4. Evaluación

### Evaluación Cuantitativa
1. Comparar los resultados de la segmentación y clasificación con las evaluaciones manuales de los radiólogos utilizando los informes médicos.

### Evaluación Clínica
2. Realizar evaluaciones visuales en un entorno clínico para determinar la calidad de las imágenes procesadas y la segmentación resultante.

