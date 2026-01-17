import os
import matplotlib.pyplot as plt

def ensure_dir(folder: str):
    """Create the output folder if it does not already exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_image_pairs(input_dir: str, mode: str, single_cc: str, single_mlo: str):
    """Return paired CC/MLO filenames from the input directory or a single pair in single mode."""
    if mode == "single":
        return [(single_cc, single_mlo)]

    image_files_cc = sorted([f for f in os.listdir(input_dir) if f.endswith(".png") and "R" in f and "CC" in f])
    image_files_mlo = sorted([f for f in os.listdir(input_dir) if f.endswith(".png") and "R" in f and "ML" in f])

    if len(image_files_cc) != len(image_files_mlo):
        print("[WARNING] Different number of CC and MLO images found!")
        print(f"CC images: {len(image_files_cc)}, MLO images: {len(image_files_mlo)}")

    return list(zip(image_files_cc, image_files_mlo))

def load_pair(input_dir: str, filename_cc: str, filename_mlo: str):
    """Load a CC/MLO image pair from disk and return them as NumPy arrays."""
    cc_image = plt.imread(os.path.join(input_dir, filename_cc))
    mlo_image = plt.imread(os.path.join(input_dir, filename_mlo))
    return cc_image, mlo_image
