import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

def intensity_ratio_propagation(image, periphery, neighborhood_size):
    """
    Correct intensity variations by scaling periphery pixels using
    a local neighborhood intensity ratio.
    """
    corrected_image = image.copy()
    rows, cols = image.shape
    half_size = neighborhood_size // 2

    for x in tqdm(range(rows), desc="Intensity ratio propagation"):
        for y in range(cols):
            if periphery[x, y]:
                xmin = max(0, x - half_size)
                xmax = min(rows, x + half_size + 1)
                ymin = max(0, y - half_size)
                ymax = min(cols, y + half_size + 1)
                neighborhood = image[xmin:xmax, ymin:ymax]

                pix = image[x, y]
                if pix == 0:
                    pix = 1e-5

                local_ratio = np.mean(neighborhood) / pix
                corrected_image[x, y] *= local_ratio

    return corrected_image


def intensity_balancing(image, skinline, ratios):
    """
    Balance image intensity using thickness-based ratios
    propagated from the breast skinline distance map.
    """
    R_values = np.log(np.array(ratios) + 1)
    Rmin = R_values.min()
    Rmax = R_values.max()

    if Rmax == Rmin:
        Rmax += 1e-5

    R_values_normalized = (R_values - Rmin) / (Rmax - Rmin)
    Rref = R_values_normalized.mean()
    RP_ref = (Rref - Rmin) / (Rmax - Rmin)

    skinline_mask = np.zeros_like(image, dtype=np.uint8)
    for x, y in skinline:
        skinline_mask[int(x), int(y)] = 1

    distance_map = distance_transform_edt(skinline_mask)
    max_distance = distance_map.max()

    balanced_image = image.copy()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            distance = distance_map[x, y]
            if distance > 0 and max_distance != 0:
                ratio_index = int((distance / max_distance) * (len(ratios) - 1))
                RP_xy = (ratios[ratio_index] - Rmin) / (Rmax - Rmin)
                balanced_image[x, y] *= (1 + (RP_ref - RP_xy))

    return balanced_image
