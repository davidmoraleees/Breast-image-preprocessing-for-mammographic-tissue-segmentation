
#Importing libraries
import pydicom
from skimage import io, filters
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_dilation, disk
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.util.shape import view_as_windows

image = plt.imread('PNG_Images/IMG001.png')
image=0.299*image[:,:,0]+0.587*image[:,:,1]+0.114*image[:,:,2] #Luma

#2.1 Breast perifery separation
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


# 2.2 Propagation of intensity ratio
distance_map = distance_transform_edt(np.logical_not(largest_component_mask))

window_size = 17
padded_image = np.pad(image, window_size // 2, mode='reflect')
windows = view_as_windows(padded_image, (window_size, window_size))
local_means = np.mean(windows, axis=(2, 3))
local_means = local_means[:image.shape[0], :image.shape[1]]

adjusted_intensity = image * local_means
final_image = adjusted_intensity * largest_component_mask

plt.figure()
plt.imshow(final_image, cmap='gray')
plt.show()


