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

#cc_corrected = intensity_ratio_propagation(cc_image, cc_bpa)
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
#No sé si el código de arriba es correcto, pero tiene algo de sentido.

#Ahora hay que dibujar unas líneas. Este código las dibuja mal, pero lo dejo por si necesitamos
#inspirarnos en este para hacerlas correctamente:

def find_skin_lines(pb):
    nipple_index = np.argmax(pb[:, 1])
    upper_skinline = pb[:nipple_index + 1]
    lower_skinline = pb[nipple_index:]
    return upper_skinline, lower_skinline

def generate_parallel_lines(upper_skinline, lower_skinline, num_lines=3):
    lines = []
    for i in range(num_lines):
        upper_idx = int(len(upper_skinline) * (i + 1) / (num_lines + 1))
        lower_idx = int(len(lower_skinline) * (i + 1) / (num_lines + 1))
        upper_point = upper_skinline[upper_idx]
        lower_point = lower_skinline[lower_idx]
        lines.append((upper_point, lower_point))
    return lines

upper_skinline, lower_skinline = find_skin_lines(mlo_pb)
parallel_lines = generate_parallel_lines(upper_skinline, lower_skinline)

plt.figure(figsize=(12, 12))
plt.imshow(mlo_image, cmap='gray')

if parallel_lines:
    for line in parallel_lines:
        rr, cc = draw.line(int(line[0][0]), int(line[0][1]), int(line[1][0]), int(line[1][1]))
        plt.plot(cc, rr, '-r', linewidth=2)

plt.title('MLO View with Parallel Lines')
plt.show()