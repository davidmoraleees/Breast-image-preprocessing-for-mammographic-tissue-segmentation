import os
import argparse
import numpy as np

from .config import (
    DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_MODE,
    DEFAULT_SINGLE_CC, DEFAULT_SINGLE_MLO,
    NEIGHBORHOOD_SIZE, KMEANS_CLUSTERS,
    OFFSET_DISTANCE, THICKEST_SHIFT
)

from .io_utils import ensure_dir, get_image_pairs, load_pair
from .plot_utils import axis_off
from .plot_utils import (
    save_periphery_plot,
    save_intensity_plot,
    save_mlo_periphery_plot,
    save_balanced_plot,
    save_clusters_plot
)
from .periphery import separate_periphery
from .intensity import intensity_ratio_propagation, intensity_balancing
from .thickness import (
    find_farthest_point_from_chest_wall,
    draw_reference_and_parallel_lines,
    calculate_length_ratios,
    apply_length_ratios,
)
from .segmentation import kmeans_segmentation


def main():
    """
    Run the full mammogram preprocessing pipeline (single or batch)
    and save all intermediate result plots.
    """
    os.environ["LOKY_MAX_CPU_COUNT"] = "2"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["single", "batch"])
    parser.add_argument("--single-cc", default=DEFAULT_SINGLE_CC)
    parser.add_argument("--single-mlo", default=DEFAULT_SINGLE_MLO)
    args = parser.parse_args()

    ensure_dir(args.output)

    pairs = get_image_pairs(args.input, args.mode, args.single_cc, args.single_mlo)

    for filename_cc, filename_mlo in pairs:
        id_image = filename_cc[:-17]

        print(f"\n==================== {id_image} ====================")
        print(f"[INFO] CC: {filename_cc}")
        print(f"[INFO] MLO: {filename_mlo}")

        print("[STEP 0] Loading images...")
        cc_image, mlo_image = load_pair(args.input, filename_cc, filename_mlo)
        print("[OK] Images loaded.")

        print("[STEP 1/5] Periphery separation...")
        cc_bpa, cc_pb = separate_periphery(cc_image)
        mlo_bpa, mlo_pb = separate_periphery(mlo_image)
        save_periphery_plot(args.output, id_image, cc_image, mlo_image, cc_bpa, mlo_bpa, cc_pb, mlo_pb)
        print("[OK] Saved periphery plot.")

        print("[STEP 2/5] Intensity ratio propagation...")
        cc_corrected = intensity_ratio_propagation(cc_image, cc_bpa, NEIGHBORHOOD_SIZE)
        mlo_corrected = intensity_ratio_propagation(mlo_image, mlo_bpa, NEIGHBORHOOD_SIZE)
        save_intensity_plot(args.output, id_image, cc_image, mlo_image, cc_corrected, mlo_corrected)
        print("[OK] Saved intensity propagation plot.")

        print("[STEP 3/5] Thickness estimation + ratios...")
        farthest_mlo, idx_mlo = find_farthest_point_from_chest_wall(mlo_pb, mlo_image.shape[1])
        mlo_pb_upper = mlo_pb[: idx_mlo + 1]
        mlo_pb_lower = mlo_pb[idx_mlo:]

        thickest_mlo = np.copy(farthest_mlo)
        thickest_mlo[1] += THICKEST_SHIFT

        num_lines = len(mlo_pb_upper)
        parallel_lines, closest_line, slope = draw_reference_and_parallel_lines(
            mlo_pb, OFFSET_DISTANCE, num_lines, thickest_mlo
        )

        if closest_line is None:
            print(f"[WARNING] No closest line for {id_image}, skipping...")
            continue

        ratios = calculate_length_ratios(parallel_lines, closest_line)
        ratios_cc = apply_length_ratios(cc_corrected, cc_pb, ratios)
        ratios_mlo = apply_length_ratios(mlo_corrected, mlo_pb, ratios)

        save_mlo_periphery_plot(args.output, id_image, mlo_bpa, mlo_pb_upper, mlo_pb_lower, farthest_mlo)
        print("[OK] Saved MLO thickness plot.")

        print("[STEP 4/5] Intensity balancing...")
        balanced_cc = intensity_balancing(ratios_cc, cc_pb, ratios)
        balanced_mlo = intensity_balancing(ratios_mlo, mlo_pb, ratios)
        save_balanced_plot(args.output, id_image, balanced_cc, balanced_mlo)
        print("[OK] Saved balanced image plot.")

        print("[STEP 5/5] KMeans segmentation...")
        clusters_cc = kmeans_segmentation(cc_image, KMEANS_CLUSTERS)
        clusters_mlo = kmeans_segmentation(mlo_image, KMEANS_CLUSTERS, ref_image=cc_image)
        clusters_cc_bal = kmeans_segmentation(balanced_cc, KMEANS_CLUSTERS, ref_image=cc_image)
        clusters_mlo_bal = kmeans_segmentation(balanced_mlo, KMEANS_CLUSTERS, ref_image=cc_image)

        save_clusters_plot(args.output, id_image, clusters_cc, clusters_mlo, clusters_cc_bal, clusters_mlo_bal)
        print("[OK] Saved clustering plot.")

        print(f"[DONE] Finished case {id_image}")


if __name__ == "__main__":
    main()
