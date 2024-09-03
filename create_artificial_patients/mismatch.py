import numpy as np
import os
from PIL import Image

# Load classifications
patient_folder = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages/MLL_232168'
probs_path = os.path.join(patient_folder, 'single_cell_probabilities.npy')
sc_probs = np.load(probs_path)

# List all images
images = sorted([f for f in os.listdir(patient_folder) if f.lower().endswith('.rgb.tif')])

# Compare lengths
print(f"Number of images: {len(images)}, Number of classification results: {len(sc_probs)}")

# Check for missing files
if len(images) != len(sc_probs):
    print("Mismatch detected. Investigating further...")

    # Print missing image files or extra entries
    for idx, image in enumerate(images):
        if idx >= len(sc_probs):
            print(f"Extra image without classification: {image}")
        else:
            try:
                # Attempt to open image to check for any issues
                img_path = os.path.join(patient_folder, image)
                img = Image.open(img_path)
            except Exception as e:
                print(f"Error opening image {image}: {e}")
