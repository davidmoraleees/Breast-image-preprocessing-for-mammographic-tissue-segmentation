
from skimage import io, filters
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_dilation, disk
from skimage.measure import label, regionprops
from skimage.util.shape import view_as_windows
from skimage.filters import sobel
from skimage.measure import find_contours


def breast_perifery_sepparation(image):
    otsu_threshold = filters.threshold_otsu(image)
    otsu_image = image > otsu_threshold

    mean_threshold = np.mean(image[otsu_image])
    mean_image = image > mean_threshold
    improved_binary_image = np.logical_or(otsu_image, mean_image)

    filled_image = binary_fill_holes(improved_binary_image)
    structuring_element = disk(1)  
    dilated_image = binary_dilation(filled_image, structuring_element)

    labeled_image, num_features = label(dilated_image, return_num=True)
    props = regionprops(labeled_image)

    largest_component = None
    for component in props:
        if largest_component is None or component.area > largest_component.area:
            largest_component = component
    largest_component_mask = (labeled_image == largest_component.label)
    return largest_component_mask


def propagation_of_intensity_ratio(image, largest_component_mask, window_size):
    padded_image = np.pad(image, window_size // 2, mode='reflect')
    windows = view_as_windows(padded_image, (window_size, window_size))
    local_means = np.mean(windows, axis=(2, 3))
    local_means = local_means[:image.shape[0], :image.shape[1]]

    adjusted_intensity = image * local_means
    final_image = adjusted_intensity * largest_component_mask
    return final_image


image_cc = plt.imread('INbreast/AllDICOMs_PNG/20586908_6c613a14b80a8591_MG_R_CC_ANON.png')
largest_component_mask_cc=breast_perifery_sepparation(image_cc)
final_image_cc=propagation_of_intensity_ratio(image_cc, largest_component_mask_cc, 17)

image_mlo=plt.imread('INbreast/AllDICOMs_PNG/20586960_6c613a14b80a8591_MG_R_ML_ANON.png')
largest_component_mask_mlo=breast_perifery_sepparation(image_mlo)
final_image_mlo=propagation_of_intensity_ratio(image_mlo, largest_component_mask_mlo, 17)




