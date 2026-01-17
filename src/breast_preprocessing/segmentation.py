import numpy as np
from sklearn.cluster import KMeans
from skimage import color

def kmeans_segmentation(image, n_clusters, ref_image=None):
    """
    Segment an image using K-Means clustering, optionally
    initialized from a reference image for consistency.
    """
    flat_image = image.reshape((-1, 1))

    if ref_image is not None:
        flat_ref_image = ref_image.reshape((-1, 1))
        initial_centers = KMeans(
            n_clusters=n_clusters, init="k-means++", n_init=1, random_state=0
        ).fit(flat_ref_image).cluster_centers_
    else:
        initial_centers = KMeans(
            n_clusters=n_clusters, init="k-means++", n_init=1, random_state=0
        ).fit(flat_image).cluster_centers_

    kmn = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, random_state=0).fit(flat_image)
    labels_image = kmn.predict(flat_image)

    clustered_image = np.reshape(labels_image, [image.shape[0], image.shape[1]]) + 1
    colored_clustered_image = color.label2rgb(
        clustered_image,
        colors=["black", "red", "blue", "yellow", "gray"],
        bg_label=0
    )
    return colored_clustered_image
