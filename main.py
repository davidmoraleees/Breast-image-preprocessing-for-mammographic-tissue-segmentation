import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, color
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm 
from sklearn.cluster import KMeans


filename_cc = '53581264_80123a24997098dc_MG_R_CC_ANON.png' #CC image of a right breast
filename_mlo = '53581237_80123a24997098dc_MG_R_ML_ANON.png' #MLO image of a right breast

id_image = filename_cc[:-17]

cc_image_path = os.path.join('INbreast/AllDICOMs_PNG', filename_cc)
mlo_image_path = os.path.join('INbreast/AllDICOMs_PNG', filename_mlo)

cc_image = plt.imread(cc_image_path)
mlo_image = plt.imread(mlo_image_path)

''''
INDEX 
1. Breast periphery separation
2. Intensity ratio propagation
3. Breast thickness estimation
4. Intensity balancing
5. Breast segmentation
'''

# 1. Breast periphery separation ------------------------------------------------------------------------------
def separate_periphery(image): # Function to separate the breast peripheral area (BPA) using Otsu thresholding
    # Threshold method 1
    otsu_thresh = filters.threshold_otsu(image) # analyzes the image histogram and calculates the optimal intensity threshold using Otsu's method.
    bpa_otsu = image > otsu_thresh # BPA's binary image using Otsu's thresholding method

    # Threshold method 2
    mean_intensity = image[bpa_otsu].mean() # Calculate the mean intensity of the breast tissue
    bpa_threshold = image > mean_intensity # BPA's binary image using the mean intensity thresholding method

    # Combination of both mehtods  
    bpa_combined = np.logical_or(bpa_otsu, bpa_threshold) # Combine the two thresholding methods to get a better separation (True if at least one of the thresholding methods considers it as part of the foreground)
    bpa_filled = morphology.remove_small_holes(bpa_combined, area_threshold=64) # Fill small holes in the binary image (for areas =< 64 pixels)
    bpa_dilated = morphology.binary_dilation(bpa_filled, morphology.square(3)) # Dilate the binary image to include boundary pixels 

    # Finding the largest region, which corresponds to the breast
    labeled_bpa = measure.label(bpa_dilated) # Label connected regions in the binary image
    regions = measure.regionprops(labeled_bpa) # Calculates the properties of the labeled regions in the image
    largest_region = max(regions, key=lambda r: r.area) # Finds the largest region, which should correspond to the breast

    # bpa_final : The final binary image contains only the largest region, which is assumed to be breast tissue.
    bpa_final = np.zeros_like(bpa_combined)
    bpa_final[labeled_bpa == largest_region.label] = 1  # Assigns a true value to the pixels that belong to the largest region in the labeled image
    
    # Finds the breast tissue contour
    pb = measure.find_contours(bpa_final, 0.5)[0] # Find the contour of the breast periphery 
    return bpa_final, pb

cc_bpa, cc_pb = separate_periphery(cc_image)
mlo_bpa, mlo_pb = separate_periphery(mlo_image)

def axis_off(): # Function to configure axis plots
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

output_dir = 'Output_images'
if not os.path.exists(output_dir): # We make sure that the output images folder exists
    os.makedirs(output_dir)  

plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original image')
axis_off()

plt.subplot(2, 2, 2)
plt.imshow(cc_bpa, cmap='gray')
plt.plot(cc_pb[:, 1], cc_pb[:, 0], '-r', linewidth=2)
plt.title('CC Peripheral area')
axis_off()

plt.subplot(2, 2, 3)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original image')
axis_off()

plt.subplot(2, 2, 4)
plt.imshow(mlo_bpa, cmap='gray')
plt.plot(mlo_pb[:, 1], mlo_pb[:, 0], '-r', linewidth=2)
plt.title('MLO Peripheral area')
axis_off()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'separate_periphery_{id_image}.png'))
plt.show()


# 2. Intensity ratio propagation ---------------------------------------------------------------------------------
def intensity_ratio_propagation(image, periphery, neighborhood_size): # Function to propagate intensity ratio for correcting intensity variations
    distance_map = distance_transform_edt(periphery) # Calculate the distance transform of the periphery
    corrected_image = image.copy()
    rows, cols = image.shape
    half_size = neighborhood_size // 2

    for y in tqdm(range(rows)):
        for x in range(cols):
            if periphery[y, x]:
                ymin = max(0, y-half_size)
                ymax = min(rows, y+half_size+1)
                xmin = max(0, x-half_size)
                xmax = min(cols, x+half_size+1)
            
                neighborhood = image[ymin:ymax, xmin:xmax] 
                local_ratio = np.mean(neighborhood) / (image[y, x] + 1e-8) #We add 1e-8 to avoid division by zero.
                corrected_image[y, x] *= local_ratio
    
    return corrected_image

cc_corrected = intensity_ratio_propagation(cc_image, cc_bpa, 17) 
mlo_corrected = intensity_ratio_propagation(mlo_image, mlo_bpa, 17)
 
plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original image')
axis_off()

plt.subplot(2, 2, 2)
plt.imshow(cc_corrected, cmap='gray')
plt.title('CC Corrected image')
axis_off()

plt.subplot(2, 2, 3)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original image')
axis_off()

plt.subplot(2, 2, 4)
plt.imshow(mlo_corrected, cmap='gray')
plt.title('MLO Corrected image')
axis_off()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'intensity_ratio_propagation_{id_image}.png'))
plt.show()


# 3. Breast thickness estimation ---------------------------------------------------------------------------------
# From now on, it is assumed that we have a right breast image, so the chest wall is placed at the right edge.
def find_furthest_point_from_chest_wall(skinline, image_width): # The point found will be close to the nipple.
    chest_wall_x = image_width - 1  # Chest wall x-coordinate (right edge of image).
    distances = chest_wall_x - skinline[:, 1] # vector containing the horizontal distances from the chest wall to each point on the skin line
    furthest_point_index = np.argmax(distances) # Find the index of the point with the maximum distance.
    furthest_point = skinline[furthest_point_index] # Returns the furthest point and its index
    return furthest_point, furthest_point_index

image_width_cc = cc_image.shape[1]
furthest_point_cc, furthest_point_index_cc = find_furthest_point_from_chest_wall(cc_pb, image_width_cc)
image_width_mlo = mlo_image.shape[1]
furthest_point_mlo, furthest_point_index_mlo = find_furthest_point_from_chest_wall(mlo_pb, image_width_mlo)

mlo_pb_upper = mlo_pb[:furthest_point_index_mlo + 1]
mlo_pb_lower = mlo_pb[furthest_point_index_mlo:]

plt.figure(figsize=(6,6))
plt.imshow(mlo_bpa, cmap='gray')
plt.plot(mlo_pb_upper[:, 1], mlo_pb_upper[:, 0], '-b', linewidth=2)
plt.plot(mlo_pb_lower[:, 1], mlo_pb_lower[:, 0], '-g', linewidth=2)
plt.plot(furthest_point_mlo[1], furthest_point_mlo[0], 'yo')
plt.title('MLO Peripheral area')
axis_off()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'MLO_peripheral_area_{id_image}.png'))
plt.show()

def find_nearest_top(skinline): # Function to find the closest point to the top edge of the image.
    top_point_index = np.argmin(skinline[:, 0])
    return skinline[top_point_index]

def find_nearest_right(skinline): #Function to find the closest point to the right edge of the image.
    right_point_index = np.argmax(skinline[:, 1])
    return skinline[right_point_index]

def calculate_slope(point1, point2): #Function to calculate the slope between two points.
    return (point2[0] - point1[0]) / (point2[1] - point1[1])

def find_intersection(skinline, slope, intercept): # Finds the point of intersection of a line with the skin line.
    for y, x in skinline:
        if np.isclose(y, slope * x + intercept, atol=1.0):
            return [y, x]
    return None

thickest_point_cc=np.copy(furthest_point_cc)
thickest_point_cc[1]+=180. # We add some distance so we are not on the furthest point, now we are on the thickest point.
thickest_point_mlo=np.copy(furthest_point_mlo)
thickest_point_mlo[1]+=180. 

def draw_reference_and_parallel_lines(image, skinline, offset_distance, num_lines, thickest_point, output_dir, id_image):
    top_reference = find_nearest_top(skinline)
    right_reference = find_nearest_right(skinline)
    
    slope = calculate_slope(top_reference, right_reference)
    intercept = top_reference[0] - slope * top_reference[1] 
    
    parallel_lines = []
    min_distance = float(10**8) # Initialization of the variable.
    closest_line = None

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    
    for i in tqdm(range(num_lines)):
        parallel_intercept = intercept - (i + 1) * offset_distance / np.cos(np.arctan(slope))
        parallel_top = find_intersection(skinline, slope, parallel_intercept)
        parallel_bottom = find_intersection(skinline[::-1], slope, parallel_intercept)
        
        if parallel_top is not None and parallel_bottom is not None:
            plt.plot([parallel_top[1], parallel_bottom[1]], [parallel_top[0], parallel_bottom[0]], '-r', linewidth=2)
            parallel_lines.append((parallel_top, parallel_bottom))

            distance = np.abs(slope * thickest_point[1] - thickest_point[0] + parallel_intercept) / np.sqrt(slope**2 + 1)
            if distance < min_distance:
                min_distance = distance
                parallel_top = [float(parallel_top[0]), float(parallel_top[1])]
                parallel_bottom = [float(parallel_bottom[0]), float(parallel_bottom[1])]
                closest_line = (parallel_top, parallel_bottom)

    plt.plot([top_reference[1], right_reference[1]], [top_reference[0], right_reference[0]], '-r', linewidth=2)
    plt.plot(top_reference[1], top_reference[0])
    plt.plot(right_reference[1], right_reference[0])
    
    plt.title('MLO Image with parallel lines')
    axis_off()

    if closest_line is not None:
        parallel_top, parallel_bottom = closest_line
        plt.subplot(1, 2, 2)
        plt.imshow(mlo_bpa, cmap='gray')
        plt.plot([parallel_top[1], parallel_bottom[1]], [parallel_top[0], parallel_bottom[0]], '-y', linewidth=2)
        plt.plot(thickest_point_mlo[1], thickest_point_mlo[0], 'yo')
        plt.title('Closest parallel line to thickest point')
        axis_off()
    else:
        print("No line found close to the thickest point.")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parallel_lines_thickest_point_{id_image}.png'))
    plt.show()

    return parallel_lines, closest_line

offset_distance = -1 #Negative value means going left.
num_lines = len(mlo_pb_upper) #We want the upper skinline to match one point of the lower skinline.
parallel_lines, closest_line = draw_reference_and_parallel_lines(mlo_bpa, mlo_pb, offset_distance, num_lines, thickest_point_mlo, output_dir, id_image)

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

if closest_line is not None:
    ratios = calculate_ratios(parallel_lines, closest_line)
else:
    ratios = []

def propagate_ratios(image, skinline, ratios):
    # Crear un mapa binario a partir de skinline
    skinline_mask = np.zeros_like(image, dtype=np.uint8)
    for y, x in skinline:
        skinline_mask[int(y), int(x)] = 1

    distance_map = distance_transform_edt(skinline_mask)
    ratios_propagated = image.copy()
    max_distance = distance_map.max()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            distance = distance_map[y, x]
            if distance > 0:
                ratio_index = int((distance / max_distance) * (len(ratios) - 1))
                ratios_propagated[y, x] *= ratios[ratio_index]

    return ratios_propagated

if ratios:
    ratios_propagated_cc = propagate_ratios(cc_corrected, cc_pb, ratios)
    ratios_propagated_mlo = propagate_ratios(mlo_corrected, mlo_pb, ratios)

    plt.figure(figsize=(6, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cc_image, cmap='gray')
    plt.title('CC Original image')
    axis_off()

    plt.subplot(2, 2, 2)
    plt.imshow(ratios_propagated_cc, cmap='gray')
    plt.title('CC Ratios propagated image')
    axis_off()

    plt.subplot(2, 2, 3)
    plt.imshow(mlo_image, cmap='gray')
    plt.title('MLO Original image')
    axis_off()

    plt.subplot(2, 2, 4)
    plt.imshow(ratios_propagated_mlo, cmap='gray')
    plt.title('MLO Ratios propagated image')
    axis_off()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ratios_propagated_{id_image}.png'))
    plt.show()
else:
    print("Ratios were not calculated due to lack of nearest reference line.")


# 4. Intensity balancing -------------------------------------------------------------------------------------------------------------
def intensity_balancing(image, skinline, ratios):
    # Calcular Rref
    R_values = np.array(ratios)
    Rmin = R_values.min()
    Rmax = R_values.max()
    Rref = R_values.mean()

    RP_ref = (Rref - Rmin) / (Rmax - Rmin)

    # Crear un mapa binario a partir de skinline
    skinline_mask = np.zeros_like(image, dtype=np.uint8)
    for y, x in skinline:
        skinline_mask[int(y), int(x)] = 1

    distance_map = distance_transform_edt(skinline_mask)
    max_distance = distance_map.max()
    balanced_image = image.copy()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            distance = distance_map[y, x]
            if distance > 0:
                ratio_index = int((distance / max_distance) * (len(ratios) - 1))
                RP_xy = (ratios[ratio_index] - Rmin) / (Rmax - Rmin)
                balanced_image[y, x] *= (1 + (RP_ref - RP_xy))

    return balanced_image

if ratios:
    balanced_cc_image = intensity_balancing(ratios_propagated_cc, cc_pb, ratios)
    balanced_mlo_image = intensity_balancing(ratios_propagated_mlo, mlo_pb, ratios)

    plt.figure(figsize=(6, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cc_image, cmap='gray')
    plt.title('CC Original image')
    axis_off()

    plt.subplot(2, 2, 2)
    plt.imshow(balanced_cc_image, cmap='gray')
    plt.title('CC Balanced image')
    axis_off()

    plt.subplot(2, 2, 3)
    plt.imshow(mlo_image, cmap='gray')
    plt.title('MLO Original image')
    axis_off()

    plt.subplot(2, 2, 4)
    plt.imshow(balanced_mlo_image, cmap='gray')
    plt.title('MLO Balanced image')
    axis_off()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'balanced_images_{id_image}.png'))
    plt.show()
else:
    print("Ratios were not calculated due to lack of nearest reference line.")


# 5. Breast segmentation  -----------------------------------------------------------------------------------------------------------
def kmeans_segmentation(image, n_clusters, ref_image=None):
    flat_image = image.reshape((-1, 1))  
    
    if ref_image is not None: # We make sure that both images start with the same centroids. Otherwise, colours are swapped.                         
        flat_ref_image = ref_image.reshape((-1, 1))
        initial_centers = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=0).fit(flat_ref_image).cluster_centers_
    else:
        initial_centers = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=0).fit(flat_image).cluster_centers_
    
    kmn = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, random_state=0).fit(flat_image)
    labels_image = kmn.predict(flat_image)
    clustered_image = np.reshape(labels_image, [image.shape[0], image.shape[1]]) + 1
    colored_clustered_image = color.label2rgb(clustered_image, colors=['black', 'red', 'darkblue', 'yellow', 'gray'], bg_label=0)
    return colored_clustered_image

colored_clustered_image_cc = kmeans_segmentation(cc_image, 5)
colored_clustered_image_mlo = kmeans_segmentation(mlo_image, 5, ref_image=cc_image)
colored_clustered_image_cc_balanced = kmeans_segmentation(balanced_cc_image, 5, ref_image=cc_image)
colored_clustered_image_mlo_balanced = kmeans_segmentation(balanced_mlo_image, 5, ref_image=cc_image)

plt.figure(figsize=(6,6))
plt.subplot(2,2,1)
plt.imshow(colored_clustered_image_cc)
plt.title('CC Unprocessed clustered image')
axis_off()

plt.subplot(2,2,2)
plt.imshow(colored_clustered_image_cc_balanced)
plt.title('CC Processed clustered image')
axis_off()

plt.subplot(2,2,3)
plt.imshow(colored_clustered_image_mlo)
plt.title('MLO Unprocessed clustered image')
axis_off()

plt.subplot(2,2,4)
plt.imshow(colored_clustered_image_mlo_balanced)
plt.title('MLO Processed clustered image')
axis_off()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'clustering_images_{id_image}.png'))
plt.show()

