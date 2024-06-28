import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, draw
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


cc_image = io.imread('INbreast/AllDICOMs_PNG/20586908_6c613a14b80a8591_MG_R_CC_ANON.png', as_gray=True)
mlo_image = io.imread('INbreast/AllDICOMs_PNG/20586960_6c613a14b80a8591_MG_R_ML_ANON.png', as_gray=True)


# Breast periphery separation ------------------------------------------------------------------------------
def separate_periphery(image):
    otsu_thresh = filters.threshold_otsu(image)
    bpa_otsu = image > otsu_thresh   
    mean_intensity = image[bpa_otsu].mean()
    bpa_threshold = image > mean_intensity    
    bpa_combined = np.logical_or(bpa_otsu, bpa_threshold)   
    bpa_filled = morphology.remove_small_holes(bpa_combined, area_threshold=64)
    bpa_dilated = morphology.binary_dilation(bpa_filled, morphology.square(3))    
    labeled_bpa = measure.label(bpa_dilated)
    regions = measure.regionprops(labeled_bpa)
    largest_region = max(regions, key=lambda r: r.area)
    bpa_final = np.zeros_like(bpa_combined)
    bpa_final[labeled_bpa == largest_region.label] = 1   
    pb = measure.find_contours(bpa_final, 0.5)[0]
    return bpa_final, pb


cc_bpa, cc_pb = separate_periphery(cc_image)
mlo_bpa, mlo_pb = separate_periphery(mlo_image)

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original Image')

plt.subplot(2, 2, 2)
plt.imshow(cc_bpa, cmap='gray')
plt.plot(cc_pb[:, 1], cc_pb[:, 0], '-r', linewidth=2)
plt.title('CC Peripheral Area')

plt.subplot(2, 2, 3)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original Image')

plt.subplot(2, 2, 4)
plt.imshow(mlo_bpa, cmap='gray')
plt.plot(mlo_pb[:, 1], mlo_pb[:, 0], '-r', linewidth=2)
plt.title('MLO Peripheral Area')

plt.show()


# Intensity ratio propagation ---------------------------------------------------------------------------------
def intensity_ratio_propagation(image, periphery):
    distance_map = distance_transform_edt(periphery)
    corrected_image = image.copy()
    rows, cols = image.shape
    neighborhood_size = 17
    half_size = neighborhood_size // 2
    
    for y in tqdm(range(rows)):
        for x in range(cols):
            if periphery[y, x]:
                ymin = max(0, y-half_size)
                ymax = min(rows, y+half_size+1)
                xmin = max(0, x-half_size)
                xmax = min(cols, x+half_size+1)
                
                neighborhood = image[ymin:ymax, xmin:xmax]
                local_ratio = np.mean(neighborhood) / (image[y, x] + 1e-8)
                corrected_image[y, x] *= local_ratio
    
    return corrected_image

#cc_corrected = intensity_ratio_propagation(cc_image, cc_bpa) Esto lo he comentado porque tarda mucho
#en compilar, y estaba haciendo otras pruebas.

#mlo_corrected = intensity_ratio_propagation(mlo_image, mlo_bpa)

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original Image')

plt.subplot(2, 2, 2)
#plt.imshow(cc_corrected, cmap='gray')
plt.title('CC Corrected Image')

plt.subplot(2, 2, 3)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original Image')

plt.subplot(2, 2, 4)
#plt.imshow(mlo_corrected, cmap='gray')
plt.title('MLO Corrected Image')

plt.show()


# Breast thickness estimation ---------------------------------------------------------------------------------
def find_furthest_point_from_chest_wall(skinline, image_width):
    chest_wall_x = image_width - 1 # Asumiendo que la pared torácica está en el borde derecho de la imagen.
    distances = chest_wall_x - skinline[:, 1]
    furthest_point_index = np.argmax(distances)
    furthest_point = skinline[furthest_point_index]
    return furthest_point, furthest_point_index


image_width_cc = cc_image.shape[1]
image_width_mlo = mlo_image.shape[1]

furthest_point_cc, furthest_point_index_cc = find_furthest_point_from_chest_wall(cc_pb, image_width_cc)
furthest_point_mlo, furthest_point_index_mlo = find_furthest_point_from_chest_wall(mlo_pb, image_width_mlo)

cc_pb_upper = cc_pb[:furthest_point_index_cc+1]
cc_pb_lower = cc_pb[furthest_point_index_cc:]

mlo_pb_upper = mlo_pb[:furthest_point_index_mlo+1]
mlo_pb_lower = mlo_pb[furthest_point_index_mlo:]

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original Image')

plt.subplot(2, 2, 2)
plt.imshow(cc_bpa, cmap='gray')
plt.plot(cc_pb_upper[:, 1], cc_pb_upper[:, 0], '-b', linewidth=2)
plt.plot(cc_pb_lower[:, 1], cc_pb_lower[:, 0], '-g', linewidth=2)
plt.plot(furthest_point_cc[1], furthest_point_cc[0], 'yo')  
plt.title('CC Peripheral Area')

plt.subplot(2, 2, 3)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original Image')

plt.subplot(2, 2, 4)
plt.imshow(mlo_bpa, cmap='gray')
plt.plot(mlo_pb_upper[:, 1], mlo_pb_upper[:, 0], '-b', linewidth=2)
plt.plot(mlo_pb_lower[:, 1], mlo_pb_lower[:, 0], '-g', linewidth=2)
plt.plot(furthest_point_mlo[1], furthest_point_mlo[0], 'yo')  
plt.title('MLO Peripheral Area')

plt.show()

#La recta no acaba de hacerse bien. Hay que acabar de afinar el código siguiente para que la recta se 
#dibuje corectamente.
def find_nearest_top(skinline):
    top_point_index = np.argmin(skinline[:, 0])
    return skinline[top_point_index]

def find_nearest_right(skinline, image_width):
    right_point_index = np.argmin(image_width - skinline[:, 1])
    return skinline[right_point_index]

top_reference = find_nearest_top(mlo_pb_upper)
right_reference = find_nearest_right(mlo_pb_lower, image_width_mlo)

if top_reference is None or right_reference is None:
    print("No se encontraron puntos de referencia válidos cerca de los bordes especificados.")
else:
    plt.figure(figsize=(12, 12))
    plt.imshow(mlo_image, cmap='gray')
    plt.plot(mlo_pb[:, 1], mlo_pb[:, 0], '-r', linewidth=2)
    plt.plot([top_reference[1], right_reference[1]], [top_reference[0], right_reference[0]], '-b', linewidth=2)
    plt.plot(furthest_point_mlo[1], furthest_point_mlo[0], 'bo') 
    plt.title('MLO Image with Reference Line')
    plt.show()

    