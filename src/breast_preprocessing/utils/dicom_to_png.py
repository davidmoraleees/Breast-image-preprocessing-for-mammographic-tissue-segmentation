import os
import argparse
import pydicom
import imageio
from tqdm import tqdm
import numpy as np


def main():
    """Convert all DICOM (.dcm) files in an input folder into normalized PNG images."""
    parser = argparse.ArgumentParser(description="Convert DICOM (.dcm) images to PNG")
    parser.add_argument("--input", required=True, help="Path to input folder containing .dcm files")
    parser.add_argument("--output", required=True, help="Path to output folder where .png files will be saved")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dicom_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".dcm")]

    if not dicom_files:
        print(f"No .dcm files found in: {input_dir}")
        return

    progress_bar = tqdm(total=len(dicom_files), desc="Conversion in progress")

    for filename in dicom_files:
        dicom_file = os.path.join(input_dir, filename)

        try:
            dicom_data = pydicom.dcmread(dicom_file)
            image = dicom_data.pixel_array.astype(np.float32)

            if image.ndim == 3:
                image = image[0]

            # Only support 2D grayscale
            if image.ndim != 2:
                raise ValueError(f"Unsupported image shape {image.shape} (expected 2D)")

            # Normalize to [0,1]
            max_val = np.max(image)
            if max_val > 0:
                image = image / max_val

            image_uint8 = (image * 255).astype(np.uint8)

            output_filename = os.path.splitext(filename)[0] + ".png"
            output_file = os.path.join(output_dir, output_filename)

            imageio.imwrite(output_file, image_uint8)

        except Exception as e:
            print(f"\n[ERROR] Failed converting {filename}: {e}")

        progress_bar.update(1)

    progress_bar.close()
    print("Conversion completed")


if __name__ == "__main__":
    main()
