from PIL import Image, UnidentifiedImageError
import os


def check_images_in_patient_folders(base_folder):
    """
    Checks all images in each patient folder within the base folder.
    Reports any corrupted or unreadable images.

    Parameters:
    - base_folder: The path to the main directory containing patient folders.
    """
    for patient_folder in os.listdir(base_folder):
        patient_folder_path = os.path.join(base_folder, patient_folder)

        # Only process directories (assuming each patient has a unique folder)
        if os.path.isdir(patient_folder_path):
            print(f"Checking images in patient folder: {patient_folder}")

            # Iterate over each image file in the patient folder
            for image_file in os.listdir(patient_folder_path):
                image_path = os.path.join(patient_folder_path, image_file)

                # Check if the file is an image based on its extension
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    try:
                        with Image.open(image_path) as img:
                            img.verify()  # Only verifies that it is a readable image
                    except (IOError, UnidentifiedImageError):
                        print(f"Corrupted or unreadable image: {image_path}")
                else:
                    print(f"Skipping non-image file: {image_path}")


# Example usage:
check_images_in_patient_folders('/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages')
