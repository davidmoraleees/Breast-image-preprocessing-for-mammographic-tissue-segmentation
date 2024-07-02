import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, exposure, color
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from tqdm import tqdm 
from skimage.segmentation import clear_border

# Load images in grayscale
cc_image = io.imread('INbreast/AllDICOMs_PNG/20586908_6c613a14b80a8591_MG_R_CC_ANON.png', as_gray=True)
mlo_image = io.imread('INbreast/AllDICOMs_PNG/20586960_6c613a14b80a8591_MG_R_ML_ANON.png', as_gray=True)


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

# Separate periphery for both CC and MLO images
cc_bpa, cc_pb = separate_periphery(cc_image)
mlo_bpa, mlo_pb = separate_periphery(mlo_image)

# Display the original and periphery-separated images
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
# Function to propagate intensity ratio for correcting intensity variations
def intensity_ratio_propagation(image, periphery):
    # Calculate the distance transform of the periphery
    distance_map = distance_transform_edt(periphery)
    corrected_image = image.copy()
    rows, cols = image.shape
    neighborhood_size = 17
    half_size = neighborhood_size // 2
    
    # Iterate over each pixel in the image
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

# Intensity ratio propagation for both images
cc_corrected = intensity_ratio_propagation(cc_image, cc_bpa) 
mlo_corrected = intensity_ratio_propagation(mlo_image, mlo_bpa)

# Display original and intensity ratio corrected images  
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original Image')

plt.subplot(2, 2, 2)
plt.imshow(cc_corrected, cmap='gray')
plt.title('CC Corrected Image')

plt.subplot(2, 2, 3)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original Image')

plt.subplot(2, 2, 4)
plt.imshow(mlo_corrected, cmap='gray')
plt.title('MLO Corrected Image')

plt.show()


# Breast thickness estimation ---------------------------------------------------------------------------------
# Function to find the furthest point from the chest wall (right side of the image)
def find_furthest_point_from_chest_wall(skinline, image_width):
    chest_wall_x = image_width - 1  # Assuming the chest wall is at the right edge of the image
    distances = chest_wall_x - skinline[:, 1]
    furthest_point_index = np.argmax(distances)
    furthest_point = skinline[furthest_point_index]
    return furthest_point, furthest_point_index

# Function to find the nearest top point in the contour
def find_nearest_top(skinline):
    top_point_index = np.argmin(skinline[:, 0])
    return skinline[top_point_index]

# Function to find the nearest bottom point in the contour
def find_nearest_bottom(skinline):
    bottom_point_index = np.argmax(skinline[:, 0])
    return skinline[bottom_point_index]

# Function to draw reference lines similar to the paper
def draw_reference_lines(image, skinline):
    furthest_point, furthest_point_index = find_furthest_point_from_chest_wall(skinline, image.shape[1])
    top_reference = find_nearest_top(skinline)
    bottom_reference = find_nearest_bottom(skinline)
    
    # Breast center is assumed to be midway between top and bottom points
    breast_center = [(top_reference[0] + bottom_reference[0]) // 2, (top_reference[1] + bottom_reference[1]) // 2]

    # Draw the image and lines
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    
    # Draw a single red line from top to bottom (through breast center)
    plt.plot([top_reference[1], bottom_reference[1]], [top_reference[0], bottom_reference[0]], '-r', linewidth=2)

    # Mark the points
    plt.plot(top_reference[1], top_reference[0], 'yo')  # Top point
    plt.plot(breast_center[1], breast_center[0], 'yo')  # Breast center
    plt.plot(bottom_reference[1], bottom_reference[0], 'yo')  # Bottom point

    plt.title('MLO Image with Reference Line')
    plt.show()

# Calculate image width for the MLO image
image_width_mlo = mlo_image.shape[1]

# Find contours for MLO image
furthest_point_mlo, furthest_point_index_mlo = find_furthest_point_from_chest_wall(mlo_pb, image_width_mlo)
mlo_pb_upper = mlo_pb[:furthest_point_index_mlo + 1]
mlo_pb_lower = mlo_pb[furthest_point_index_mlo:]

# Draw reference lines on the MLO image
draw_reference_lines(mlo_image, mlo_pb)

# Breast segmentation  -----------------------------------------------------------------------------------------------------------
# Segmentation based on intensity and geometric features
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
    # Create an RGB version of the grayscale image
    image_rgb = color.gray2rgb(image)
    # Create a color overlay where the segmentation is highlighted
    overlay = color.label2rgb(segmentation, image_rgb, colors=['red'], alpha=0.3, bg_label=0, bg_color=None)
    return overlay

# Segment breast tissue in both images
cc_segmented = segment_breast_tissue(cc_image)
mlo_segmented = segment_breast_tissue(mlo_image)

# Create overlays
cc_overlay = overlay_segmentation(cc_image, cc_segmented)
mlo_overlay = overlay_segmentation(mlo_image, mlo_segmented)

# Display the original, segmented, and overlay images
plt.figure(figsize=(16, 16))

# CC Image
plt.subplot(2, 3, 1)
plt.imshow(cc_image, cmap='gray')
plt.title('CC Original Image')

plt.subplot(2, 3, 3)
plt.imshow(cc_segmented, cmap='gray')
plt.title('CC Segmented Image')

plt.subplot(2, 3, 5)
plt.imshow(cc_overlay)
plt.title('CC Image with Segmentation Overlay')

# MLO Image
plt.subplot(2, 3, 2)
plt.imshow(mlo_image, cmap='gray')
plt.title('MLO Original Image')

plt.subplot(2, 3, 4)
plt.imshow(mlo_segmented, cmap='gray')
plt.title('MLO Segmented Image')

plt.subplot(2, 3, 6)
plt.imshow(mlo_overlay)
plt.title('MLO Image with Segmentation Overlay')

plt.tight_layout()
plt.show()

# Function to find the furthest point from the chest wall (right side of the image)
def find_furthest_point_from_chest_wall(skinline, image_width):
    chest_wall_x = image_width - 1  # Assuming the chest wall is at the right edge of the image
    distances = chest_wall_x - skinline[:, 1]
    furthest_point_index = np.argmax(distances)
    furthest_point = skinline[furthest_point_index]
    return furthest_point, furthest_point_index

# Function to find the nearest top point in the contour
def find_nearest_top(skinline):
    top_point_index = np.argmin(skinline[:, 0])
    return skinline[top_point_index]

# Function to find the nearest bottom point in the contour
def find_nearest_bottom(skinline):
    bottom_point_index = np.argmax(skinline[:, 0])
    return skinline[bottom_point_index]

# Function to draw reference lines similar to the paper
def draw_reference_lines(image, skinline):
    furthest_point, furthest_point_index = find_furthest_point_from_chest_wall(skinline, image.shape[1])
    top_reference = find_nearest_top(skinline)
    bottom_reference = find_nearest_bottom(skinline)
    
    # Breast center is assumed to be midway between top and bottom points
    breast_center = [(top_reference[0] + bottom_reference[0]) // 2, (top_reference[1] + bottom_reference[1]) // 2]

    # Draw the image and lines
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    
    # Draw a single red line from top to bottom (through breast center)
    plt.plot([top_reference[1], bottom_reference[1]], [top_reference[0], bottom_reference[0]], '-r', linewidth=2)

    # Mark the points
    plt.plot(top_reference[1], top_reference[0], 'yo')  # Top point
    plt.plot(breast_center[1], breast_center[0], 'yo')  # Breast center
    plt.plot(bottom_reference[1], bottom_reference[0], 'yo')  # Bottom point

    plt.title('MLO Image with Reference Line')
    plt.show()

# Calculate image width for the MLO image
image_width_mlo = mlo_image.shape[1]

# Find contours for MLO image
furthest_point_mlo, furthest_point_index_mlo = find_furthest_point_from_chest_wall(mlo_pb, image_width_mlo)
mlo_pb_upper = mlo_pb[:furthest_point_index_mlo + 1]
mlo_pb_lower = mlo_pb[furthest_point_index_mlo:]

# Draw reference lines on the MLO image
draw_reference_lines(mlo_image, mlo_pb)