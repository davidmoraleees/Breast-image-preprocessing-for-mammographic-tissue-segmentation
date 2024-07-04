import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, exposure, color
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from tqdm import tqdm 
from skimage.segmentation import clear_border

cc_image = io.imread('INbreast/AllDICOMs_PNG/50997304_9054942f7be52dd9_MG_R_CC_ANON.png', as_gray=True)
mlo_image = io.imread('INbreast/AllDICOMs_PNG/50997250_9054942f7be52dd9_MG_R_ML_ANON.png', as_gray=True)


# Breast periphery separation ------------------------------------------------------------------------------
# Function to separate the breast periphery using thresholding and morphological operations
def separate_periphery(image):
    # Apply Otsu's threshold to separate the background and the breast tissue
    otsu_thresh = filters.threshold_otsu(image)
    bpa_otsu = image > otsu_thresh   

    # Calculate the mean intensity of the breast tissue
    mean_intensity = image[bpa_otsu].mean()
    bpa_threshold = image > mean_intensity    

    # Combine the two thresholding methods to get a better separation
    bpa_combined = np.logical_or(bpa_otsu, bpa_threshold)   

    # Fill small holes in the binary image
    bpa_filled = morphology.remove_small_holes(bpa_combined, area_threshold=64)

    # Dilate the binary image to include boundary pixels
    bpa_dilated = morphology.binary_dilation(bpa_filled, morphology.square(3))    

    # Label connected regions in the binary image
    labeled_bpa = measure.label(bpa_dilated)
    regions = measure.regionprops(labeled_bpa)

    # Find the largest region, which should correspond to the breast
    largest_region = max(regions, key=lambda r: r.area)
    bpa_final = np.zeros_like(bpa_combined)
    bpa_final[labeled_bpa == largest_region.label] = 1  

    # Find the contour of the breast periphery 
    pb = measure.find_contours(bpa_final, 0.5)[0]
    return bpa_final, pb

cc_bpa, cc_pb = separate_periphery(cc_image)
mlo_bpa, mlo_pb = separate_periphery(mlo_image)

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original Image')

plt.subplot(2, 2, 2)
plt.imshow(cc_image, cmap='gray')
plt.plot(cc_pb[:, 1], cc_pb[:, 0], '-r', linewidth=2)
plt.title('CC Peripheral Area')

plt.subplot(2, 2, 3)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original Image')

plt.subplot(2, 2, 4)
plt.imshow(mlo_image, cmap='gray')
plt.plot(mlo_pb[:, 1], mlo_pb[:, 0], '-r', linewidth=2)
plt.title('MLO Peripheral Area')

plt.show()


# Intensity ratio propagation ---------------------------------------------------------------------------------
# Function to propagate intensity ratio for correcting intensity variations
def intensity_ratio_propagation(image, periphery):
    # Calculate the distance transform of the periphery
    distance_map = distance_transform_edt(periphery)
    corrected_image = image.copy()
    rows, cols = image.shape
    neighborhood_size = 17
    half_size = neighborhood_size // 2
    
    for y in tqdm(range(rows)):
        for x in range(cols):
            if periphery[y, x]:
                # Define the neighborhood region
                ymin = max(0, y-half_size)
                ymax = min(rows, y+half_size+1)
                xmin = max(0, x-half_size)
                xmax = min(cols, x+half_size+1)
                
                # Calculate the local intensity ratio and apply it to the pixel
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
def find_furthest_point_from_chest_wall(skinline, image_width):
    chest_wall_x = image_width - 1  # Assuming the chest wall is at the right edge of the image
    distances = chest_wall_x - skinline[:, 1]
    furthest_point_index = np.argmax(distances)
    furthest_point = skinline[furthest_point_index]
    return furthest_point, furthest_point_index

def find_nearest_top(skinline):
    top_point_index = np.argmin(skinline[:, 0])
    return skinline[top_point_index]

def find_nearest_right(skinline):
    right_point_index = np.argmax(skinline[:, 1])
    return skinline[right_point_index]

def calculate_slope(point1, point2):
    return (point2[0] - point1[0]) / (point2[1] - point1[1])

def find_intersection(skinline, slope, intercept):
    for y, x in skinline:
        if np.isclose(y, slope * x + intercept, atol=1.0):
            return [y, x]
    return None

def draw_reference_and_parallel_lines(image, skinline, offset_distance, num_lines, thickest_point):
    top_reference = find_nearest_top(skinline)
    right_reference = find_nearest_right(skinline)
    
    slope = calculate_slope(top_reference, right_reference)
    intercept = top_reference[0] - slope * top_reference[1]
    
    parallel_lines = []
    min_distance = float('inf')
    closest_line = None

    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    
    for i in tqdm(range(num_lines)):
        parallel_intercept = intercept - (i + 1) * offset_distance / np.cos(np.arctan(slope))
        parallel_top = find_intersection(skinline, slope, parallel_intercept)
        parallel_bottom = find_intersection(skinline[::-1], slope, parallel_intercept)
        
        if parallel_top is not None and parallel_bottom is not None:
            plt.plot([parallel_top[1], parallel_bottom[1]], [parallel_top[0], parallel_bottom[0]], '-r', linewidth=2)
            parallel_lines.append((float(parallel_top[1]), float(parallel_bottom[1])))

            distance = np.abs(slope * thickest_point[1] - thickest_point[0] + parallel_intercept) / np.sqrt(slope**2 + 1)
            if distance < min_distance:
                min_distance = distance
                parallel_top[0]=float(parallel_top[0])
                parallel_top[1]=float(parallel_top[1])
                parallel_bottom[0]=float(parallel_bottom[0])
                parallel_bottom[1]=float(parallel_bottom[1])
                closest_line = (parallel_top, parallel_bottom)

    plt.plot([top_reference[1], right_reference[1]], [top_reference[0], right_reference[0]], '-r', linewidth=2)
    plt.plot(top_reference[1], top_reference[0])
    plt.plot(right_reference[1], right_reference[0])
    
    plt.title('MLO Image with Reference and Parallel Lines')
    plt.show()

    return parallel_lines, closest_line

image_width_mlo = mlo_image.shape[1]
furthest_point_mlo, furthest_point_index_mlo = find_furthest_point_from_chest_wall(mlo_pb, image_width_mlo)
mlo_pb_upper = mlo_pb[:furthest_point_index_mlo + 1]
mlo_pb_lower = mlo_pb[furthest_point_index_mlo:]

plt.figure()
plt.imshow(mlo_image, cmap='gray')
plt.plot(mlo_pb_upper[:, 1], mlo_pb_upper[:, 0], '-b', linewidth=2)
plt.plot(mlo_pb_lower[:, 1], mlo_pb_lower[:, 0], '-g', linewidth=2)
plt.title('MLO Peripheral Area')
plt.show()

offset_distance = -1 #Negative value means going left.
num_lines = len(mlo_pb_upper) #We want the upper skinline to match one point of the lower skinline.
furthest_point_mlo[1]+=180. #We add some distance so we are not on the furthest point, now we are on the thickest point.
thickest_point_mlo=np.copy(furthest_point_mlo)

parallel_lines, closest_line = draw_reference_and_parallel_lines(mlo_image, mlo_pb, offset_distance, num_lines, thickest_point_mlo)
print(closest_line)
if closest_line is not None:
    parallel_top, parallel_bottom = closest_line
    plt.figure()
    plt.imshow(mlo_image, cmap='gray')
    plt.plot([parallel_top[1], parallel_bottom[1]], [parallel_top[0], parallel_bottom[0]], '-y', linewidth=2)
    plt.plot(thickest_point_mlo[1], thickest_point_mlo[0], 'yo')
    plt.title('Closest Parallel Line to Thickest Point')
    plt.show()
else:
    print("No line found close to the thickest point.")

image_width_cc = cc_image.shape[1]
furthest_point_cc, furthest_point_index_cc = find_furthest_point_from_chest_wall(cc_pb, image_width_cc)
furthest_point_cc[1]+=180. #We add some distance so we are not on the furthest point, now we are on the thickest point.
thickest_point_cc=np.copy(furthest_point_cc)

plt.figure()
plt.imshow(cc_image, cmap='gray')
plt.plot(thickest_point_cc[1], thickest_point_cc[0], 'yo')  
plt.show()

plt.figure()
plt.imshow(mlo_image, cmap='gray')
plt.plot(thickest_point_mlo[1], thickest_point_mlo[0], 'yo')  
plt.show()





#Esto aún no funciona bien.
def calculate_length(line):
    return np.sqrt((line[1][1] - line[0][1])**2 + (line[1][0] - line[0][0])**2)

def calculate_ratios(parallel_lines, reference_line):
    reference_length = calculate_length(reference_line)
    ratios = []
    for line in parallel_lines:
        line_length = calculate_length(line)
        ratio = line_length / reference_length
        ratios.append(ratio)
    return ratios

ratios = calculate_ratios(parallel_lines, closest_line)

def propagate_ratios(image, skinline, ratios):
    distance_map = distance_transform_edt(skinline)
    corrected_image = image.copy()
    max_distance = distance_map.max()
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            distance = distance_map[y, x]
            if distance > 0:
                ratio_index = int((distance / max_distance) * (len(ratios) - 1))
                corrected_image[y, x] *= ratios[ratio_index]
    
    return corrected_image

corrected_cc_image = propagate_ratios(cc_image, cc_bpa, ratios)

plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('Original CC Image')

plt.subplot(1, 2, 2)
plt.imshow(corrected_cc_image, cmap='gray')
plt.title('Corrected CC Image')

plt.show()

def intensity_balancing(image, ratios, skinline):
    R = np.array(ratios)
    R_min = R.min()
    R_max = R.max()
    R_normalized = (R - R_min) / (R_max - R_min)
    
    R_ref = R_normalized.mean()

    distance_map = distance_transform_edt(skinline)
    corrected_image = image.copy()
    max_distance = distance_map.max()
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            distance = distance_map[y, x]
            if distance > 0:
                ratio_index = int((distance / max_distance) * (len(R_normalized) - 1))
                RP = R_normalized[ratio_index]
                corrected_image[y, x] *= (1 + (R_ref - RP))
    
    return corrected_image

balanced_cc_image = intensity_balancing(corrected_cc_image, ratios, cc_bpa)

plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.imshow(corrected_cc_image, cmap='gray')
plt.title('Corrected CC Image')

plt.subplot(1, 2, 2)
plt.imshow(balanced_cc_image, cmap='gray')
plt.title('Balanced CC Image')

plt.show()














# Breast segmentation  -----------------------------------------------------------------------------------------------------------

#Esto aún no funciona bien.
def segment_breast_tissue(image):
    # Apply histogram equalization for better contrast
    equalized_image = exposure.equalize_hist(image)
    
    # Apply Otsu's threshold to separate the background and the breast tissue
    otsu_thresh = filters.threshold_otsu(equalized_image)
    binary_image = equalized_image > otsu_thresh
    
    # Remove artifacts connected to the image border
    cleared_image = clear_border(binary_image)
    
    # Remove small objects and fill small holes
    cleaned_image = morphology.remove_small_objects(cleared_image, min_size=500)
    filled_image = binary_fill_holes(cleaned_image)
    
    # Label the connected regions
    labeled_image = measure.label(filled_image)
    regions = measure.regionprops(labeled_image)
    
    # Find the largest region, assuming it's the breast tissue
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        breast_mask = np.zeros_like(filled_image)
        breast_mask[labeled_image == largest_region.label] = 1
    else:
        breast_mask = np.zeros_like(filled_image)
    
    return breast_mask

# Function to overlay segmentation on the original image
def overlay_segmentation(image, segmentation):
    image_rgb = color.gray2rgb(image)
    # Create a color overlay where the segmentation is highlighted
    overlay = color.label2rgb(segmentation, image_rgb, colors=['red'], alpha=0.3, bg_label=0, bg_color=None)
    return overlay

# Segment breast tissue in both images
#cc_segmented = segment_breast_tissue(cc_image)
#mlo_segmented = segment_breast_tissue(mlo_image)

# Create overlays
#cc_overlay = overlay_segmentation(cc_image, cc_segmented)
#mlo_overlay = overlay_segmentation(mlo_image, mlo_segmented)

plt.figure(figsize=(16, 9))

plt.subplot(2, 3, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original Image')

plt.subplot(2, 3, 3)
#plt.imshow(cc_segmented, cmap='gray')
plt.title('CC Segmented Image')

plt.subplot(2, 3, 5)
#plt.imshow(cc_overlay)
plt.title('CC Image with Segmentation Overlay')

plt.subplot(2, 3, 2)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original Image')

plt.subplot(2, 3, 4)
#plt.imshow(mlo_segmented, cmap='gray')
plt.title('MLO Segmented Image')

plt.subplot(2, 3, 6)
#plt.imshow(mlo_overlay)
plt.title('MLO Image with Segmentation Overlay')

plt.tight_layout()
plt.show()

