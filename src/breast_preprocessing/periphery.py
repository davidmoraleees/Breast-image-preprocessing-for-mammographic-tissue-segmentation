import numpy as np
from skimage import filters, morphology, measure

def separate_periphery(image):
    """
    Extract the breast peripheral area mask and its
    contour using thresholding and morphological cleanup.
    """
    otsu_thresh = filters.threshold_otsu(image)
    bpa_otsu = image > otsu_thresh
    mean_intensity = image[bpa_otsu].mean()
    bpa_threshold = image > mean_intensity
    bpa_combined = np.logical_or(bpa_otsu, bpa_threshold)
    bpa_filled = morphology.remove_small_holes(bpa_combined, max_size=64)
    footprint = morphology.footprint_rectangle((3, 3))
    bpa_dilated = morphology.dilation(bpa_filled, footprint)

    labeled_bpa = measure.label(bpa_dilated)
    regions = measure.regionprops(labeled_bpa)

    largest_region = max(regions, key=lambda r: r.area)

    bpa_final = np.zeros_like(bpa_combined)
    bpa_final[labeled_bpa == largest_region.label] = 1

    pb = measure.find_contours(bpa_final, 0.5)[0]
    return bpa_final, pb
