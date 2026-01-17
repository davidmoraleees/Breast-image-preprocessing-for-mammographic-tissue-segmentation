import matplotlib.pyplot as plt
import numpy as np
import os


def axis_off():
    """Hide axis ticks and labels for cleaner image plots."""
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    left=False, right=False, labelbottom=False, labelleft=False)


def _save(fig_name: str, output_dir: str, id_image: str):
    """Save the current Matplotlib figure to disk and close it to free memory."""
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{fig_name}_{id_image}.png"), bbox_inches="tight")
    plt.close()


def save_periphery_plot(output_dir, id_image, cc_image, mlo_image, cc_bpa, mlo_bpa, cc_pb, mlo_pb):
    """Save a 2x2 figure showing original images and extracted breast periphery masks/contours."""
    plt.figure(figsize=(6, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(cc_image, cmap="gray")
    plt.title("CC Original image")
    axis_off()

    plt.subplot(2, 2, 2)
    plt.imshow(cc_bpa, cmap="gray")
    plt.plot(cc_pb[:, 1], cc_pb[:, 0], "-r", linewidth=2)
    plt.title("CC Peripheral area")
    axis_off()

    plt.subplot(2, 2, 3)
    plt.imshow(mlo_image, cmap="gray")
    plt.title("MLO Original image")
    axis_off()

    plt.subplot(2, 2, 4)
    plt.imshow(mlo_bpa, cmap="gray")
    plt.plot(mlo_pb[:, 1], mlo_pb[:, 0], "-r", linewidth=2)
    plt.title("MLO Peripheral area")
    axis_off()

    _save("separate_periphery", output_dir, id_image)


def save_intensity_plot(output_dir, id_image, cc_image, mlo_image, cc_corrected, mlo_corrected):
    """Save a 2x2 figure comparing intensity-corrected images with their absolute differences."""
    plt.figure(figsize=(6, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(cc_corrected, cmap="gray")
    plt.title("CC Corrected image")
    axis_off()

    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(cc_image - cc_corrected), cmap="gray")
    plt.title("CC Difference")
    axis_off()

    plt.subplot(2, 2, 3)
    plt.imshow(mlo_corrected, cmap="gray")
    plt.title("MLO Corrected image")
    axis_off()

    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(mlo_image - mlo_corrected), cmap="gray")
    plt.title("MLO Difference")
    axis_off()

    _save("intensity_ratio_propagation", output_dir, id_image)


def save_mlo_periphery_plot(output_dir, id_image, mlo_bpa, mlo_pb_upper, mlo_pb_lower, farthest_mlo):
    """Save a figure showing MLO upper/lower contour split and the farthest point from chest wall."""
    plt.figure(figsize=(3, 3))

    plt.imshow(mlo_bpa, cmap="gray")
    plt.plot(mlo_pb_upper[:, 1], mlo_pb_upper[:, 0], "-b", linewidth=3)
    plt.plot(mlo_pb_lower[:, 1], mlo_pb_lower[:, 0], "-g", linewidth=3)
    plt.plot(farthest_mlo[1], farthest_mlo[0], "yo")
    plt.title("MLO Peripheral area")
    axis_off()

    _save("MLO_peripheral_area", output_dir, id_image)


def save_balanced_plot(output_dir, id_image, balanced_cc, balanced_mlo):
    """Save a side-by-side figure of balanced CC and MLO images after intensity normalization."""
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(balanced_cc, cmap="gray")
    plt.title("CC Balanced image")
    axis_off()

    plt.subplot(1, 2, 2)
    plt.imshow(balanced_mlo, cmap="gray")
    plt.title("MLO Balanced image")
    axis_off()

    _save("balanced_images", output_dir, id_image)


def save_clusters_plot(output_dir, id_image, clusters_cc, clusters_mlo, clusters_cc_bal, clusters_mlo_bal):
    """Save a 2x2 figure comparing K-Means clustering before and after preprocessing."""
    plt.figure(figsize=(6, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(clusters_cc)
    plt.title("CC Unprocessed clusters")
    axis_off()

    plt.subplot(2, 2, 3)
    plt.imshow(clusters_mlo)
    plt.title("MLO Unprocessed clusters")
    axis_off()

    plt.subplot(2, 2, 2)
    plt.imshow(clusters_cc_bal)
    plt.title("CC Processed clusters")
    axis_off()

    plt.subplot(2, 2, 4)
    plt.imshow(clusters_mlo_bal)
    plt.title("MLO Processed clusters")
    axis_off()

    _save("clustering_images", output_dir, id_image)
